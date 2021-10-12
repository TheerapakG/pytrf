"""
Ported partial implementation for TRF format
Initial implementation is written in C++
"""

from abc import ABC, abstractmethod
import cbor2
from collections import defaultdict
from enum import IntEnum
from functools import lru_cache
from io import FileIO, SEEK_CUR, SEEK_SET
import lzma
from os import PathLike
from pathlib import Path, PurePosixPath
import shutil
import sys
from tempfile import TemporaryFile
from typing import Any, Optional, Type, Union
import zlib

assert sys.version_info[0] == 3
if sys.version_info[1] < 9:
    from typing import Dict, List
else:
    Dict = dict
    List = list

@lru_cache
def machine_int_size():
    # encode reasonable max int size
    sz = sys.maxsize * 2 + 2
    i_sz = 0
    while (sz >> 8) > 1:
        sz >>= 8
        i_sz += 1
    else:
        i_sz += 1
    return i_sz

def read_exact_size(readable: FileIO, sz: int):
    buf = b''
    while len(buf) < sz:
        buf += readable.read(sz-len(buf))
    return buf


class FileSystemObjectType(IntEnum):
    FOLDER = 0
    FILE = 1


class FileSystemAbstractObject(ABC):
    def __init__(self, name: str, *args, **kwargs) -> None:
        self.name = name
        self._parent = kwargs.pop('parent', None)
        super().__init__(*args, **kwargs)

    @property
    def parent(self):
        return self._parent

    @property
    @abstractmethod
    def meta(self):
        return self._meta

    @property
    @abstractmethod
    def type(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplementedError()


class FileSystemWriter(ABC):
    @abstractmethod
    def data(self, max_length=-1):
        raise NotImplementedError()

    @property
    @abstractmethod
    def require_offset(self) -> Dict['FileSystemObject', List[int]]:
        raise NotImplementedError()


class FileSystemObject(FileSystemAbstractObject):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

    @abstractmethod
    def copy(self):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def instantiate_from_proxy(cls, proxy: 'FileSystemProxyObject') -> 'FileSystemObject':
        raise NotImplementedError()

    @abstractmethod
    def writer(self) -> FileSystemWriter:
        raise NotImplementedError()


class BlockData:
    def __init__(self, file: FileIO, base_offset, offset, int_size, _already_seek=False) -> None:
        self.file = file
        self.base_offset = base_offset
        self.offset = offset
        self.int_size = int_size
        if not _already_seek:
            file.seek(base_offset+offset, SEEK_SET)

        next_block = int.from_bytes(read_exact_size(file, int_size), 'little') # block continuation offset from here
        self.block_length = int.from_bytes(read_exact_size(file, int_size), 'little') # block length
        self.data_start = base_offset+offset+2*int_size
        
        if next_block == 0:
            self.next_block = None
        else:
            file.seek(next_block-int_size, SEEK_CUR)
            self.next_block = BlockData(file, base_offset, offset+int_size+next_block, int_size, _already_seek=True)


class BlockDataReader:
    def __init__(self, blockdata: BlockData, seek=True) -> None:
        self._bdata = blockdata
        self._file = self._bdata.file
        if seek:
            self._file.seek(self._bdata.data_start, SEEK_SET)
        self._prev_tell = self._file.tell()
        self._pos = 0

    @property
    def int_size(self):
        return self._bdata.int_size

    def read(self, max_length=-1):
        if self._bdata:
            if self._pos == self._bdata.block_length:
                self._pos = 0
                self._bdata = self._bdata.next_block
                while self._bdata.block_length == 0:
                    self._bdata = self._bdata.next_block
                if self._bdata:
                    self._file.seek(self._bdata.data_start, SEEK_SET)
                else:
                    return b''
            else:
                self._file.seek(self._prev_tell)
        else:
            return b''

        if max_length and max_length > 0:
            data = self._file.read(min(max_length, self._bdata.block_length-self._pos))
            self._prev_tell = self._file.tell()
            self._pos += len(data)
            return data
        else:
            data = read_exact_size(self._file, self._bdata.block_length-self._pos)
            self._bdata = self._bdata.next_block
            while self._bdata:
                self._file.seek(self._bdata.data_start, SEEK_SET)
                data += read_exact_size(self._file, self._bdata.block_length)
                self._bdata = self._bdata.next_block
            return data

    def copy(self):
        n_reader = BlockDataReader(self._bdata, seek=False)
        n_reader._prev_tell = self._prev_tell
        n_reader._pos = self._pos
        return n_reader


class ReadLengthBlockDataReader:
    def __init__(self, reader: BlockDataReader, max_length=-1):
        self._reader = reader
        self._max_length = max_length

    @property
    def unlimited_reader(self):
        return self._reader

    @property
    def int_size(self):
        return self._reader.int_size

    def read(self):
        return self._reader.read(self._max_length)


class FileSystemProxyObject(FileSystemAbstractObject):
    def __init__(self, blockdata: BlockData, *args, max_length=-1, **kwargs) -> None:
        self.blockdata = blockdata
        self._reader = BlockDataReader(blockdata)
        self._max_length = max_length
        name = b''
        while True:
            r = self._reader.read(1)
            if r != b'\xFF':
                name += r
            else:
                break
        super().__init__(name.decode('utf-8'), *args, **kwargs)
        print(name.decode('utf-8'), blockdata.offset)
        self._type = FileSystemObjectType(int.from_bytes(self._reader.read(1), 'little'))
        meta_size = int.from_bytes(read_exact_size(self._reader, self._reader.int_size), 'little')
        self._meta = cbor2.loads(read_exact_size(self._reader, meta_size))
    
    @property
    def meta(self):
        return self._meta

    @property
    def type(self):
        return self._type

    def reader(self):
        return ReadLengthBlockDataReader(self._reader.copy(), self._max_length)

    def load(self):
        fo = filesystemobjects[self._type].instantiate_from_proxy(self)
        return fo
        

class Folder(FileSystemObject):
    _child: List[FileSystemAbstractObject]
    _cache: Dict[str, Optional[FileSystemObject]]

    def __init__(self, name: str, *args, **kwargs) -> None:
        self._child = kwargs.pop('child', list())
        for child in self._child:
            child._parent = self
        self._cache = {f.name: None for f in self._child}
        self._meta = kwargs.pop('meta', None)
        super().__init__(name, *args, **kwargs)
    
    @property
    def meta(self):
        return self._meta

    @property
    def type(self):
        return FileSystemObjectType.FOLDER

    def copy(self):
        childs = [child.copy() for child in self.childs]
        return Folder(self.name, child=childs, meta=self.meta)
        
    @classmethod
    def instantiate_from_proxy(cls, proxy: 'FileSystemProxyObject') -> 'Folder':
        reader = proxy.reader()
        int_size = reader.int_size
        print('load folder', proxy.name)
        print('data at', reader._reader._prev_tell-proxy.blockdata.base_offset)
        data = reader.read()
        childs = list()
        while data != b'':
            if len(data) < int_size:
                data += reader.read()
            else:
                childs.append(int.from_bytes(data[:int_size], 'little'))
                data = data[int_size:]
        print('found content offsets: ', childs)
        childs = [
            FileSystemProxyObject(
                BlockData(
                proxy.blockdata.file, 
                proxy.blockdata.base_offset, 
                child, 
                int_size
                ), 
                max_length=proxy._max_length
            ) 
            for child in childs
        ]
        return cls(proxy.name, child=childs, meta=proxy.meta, parent=proxy.parent)

    def writer(self) -> FileSystemWriter:
        class Writer(FileSystemWriter):
            def __init__(self, folder: 'Folder'):
                self._folder = folder

            def data(self, max_length=-1):
                i_sz = machine_int_size()
                if not self._folder._child:
                    return b''
                if max_length and max_length > 0:
                    count = len(self._folder._child)*i_sz
                    for i in range(0, count, max_length):
                        if i + max_length < count:
                            yield b'\x00'*max_length
                        else:
                            yield b'\x00'*(count-i)
                else:
                    yield b'\x00'*(len(self._folder._child)*i_sz)

            @property
            def require_offset(self) -> Dict[FileSystemObject, List[int]]:
                i_sz = machine_int_size()
                return {child:[i*i_sz] for i, child in enumerate(self._folder.childs)}

        return Writer(self)

    def _force_load_child(self, idx, *, f=None):
        f = self._child[idx] if not f else f
        if isinstance(f, FileSystemProxyObject):
            f = f.load()
            self._child[idx] = f
            self._cache[f.name] = f
        return f

    @property
    def childs(self):
        for i in range(len(self._child)):
            yield self._force_load_child(i)

    def __getitem__(self, path: Union[str, PathLike]) -> FileSystemObject:
        path = PurePosixPath(path)
        parts = path.parts
        if len(parts) == 0:
            return self
        if parts[0] == '/':
            return self[path.relative_to(parts[0])]
        fso = self._cache[parts[0]]
        if fso:
            if fso.name == parts[0]:
                return fso
            else:
                del self._cache[parts[0]]
        fst = next(((i, f) for i, f in enumerate(self._child) if f.name == parts[0]), None)
        if fst:
            i, fso = fst
            fso = self._force_load_child(i, f=fso)
            return fso if len(parts) == 1 else fso[path.relative_to(parts[0])]
        else:
            raise KeyError('path not in folder')

    def add(self, fsobject: FileSystemObject):
        try:
            fso = self._cache[fsobject.name]
            if fso.name == fsobject.name:
                raise Exception('name already exists')
        except KeyError:
            pass

        if fsobject.parent:
            fsobject = fsobject.copy()        
        self._child.append(fsobject)
        self._cache[fsobject.name] = fsobject
        fsobject._parent = self

    def remove(self, name:str):
        self._child = [f for f in self._child if f.name != name]
        try:
            del self._cache[name]
        except KeyError:
            pass


class CompressionType(IntEnum):
    NONE = 0
    LZMA = 1
    DEFLATE = 2
    ZSTD = 3
    BROTLI = 4


class CompressedDataGenerator(ABC):
    @abstractmethod
    def compress_gen(self, max_length=-1):
        raise NotImplementedError()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplementedError()


class Compressor(CompressedDataGenerator):
    def __init__(self, data_gen, meta_key=set()):
        super().__init__()
        self._data_gen = data_gen
        self._meta_req = meta_key

    @property
    def data_gen(self):
        return self._data_gen


class DecompressedDataGenerator(ABC):
    @abstractmethod
    def decompress_gen(self, max_length=-1):
        raise NotImplementedError()


class Decompressor(DecompressedDataGenerator):
    def __init__(self, compressed_data_gen):
        self._compressed_data_gen = compressed_data_gen

    @property
    def compressed_data_gen(self):
        return self._compressed_data_gen


class NoneCompressor(Compressor):
    def __init__(self, data_gen, meta_key=set()):
        super().__init__(data_gen, meta_key=meta_key)

    def compress_gen(self, max_length=-1):
        return self.data_gen(max_length=max_length)

    @property
    def meta(self):
        return dict()


class NoneDecompressor(Decompressor):
    def __init__(self, compressed_data_gen):
        super().__init__(compressed_data_gen)

    def decompress_gen(self, max_length=-1):
        return self.compressed_data_gen(max_length=max_length)


class LZMAFilter(IntEnum):
    NONE = 0
    X86 = 1
    IA64 = 2
    ARM = 3
    ARMTHUMB = 4
    POWERPC = 5
    SPARC = 6


lzma_filters = {
    LZMAFilter.NONE: [],
    LZMAFilter.X86: [{'id': lzma.FILTER_X86}],
    LZMAFilter.IA64: [{'id': lzma.FILTER_IA64}],
    LZMAFilter.ARM: [{'id': lzma.FILTER_ARM}],
    LZMAFilter.ARMTHUMB: [{'id': lzma.FILTER_ARMTHUMB}],
    LZMAFilter.POWERPC: [{'id': lzma.FILTER_POWERPC}],
    LZMAFilter.SPARC: [{'id': lzma.FILTER_SPARC}],
} # type: Dict[LZMAFilter, List[Dict[str, Any]]]


class LZMACompressor(Compressor):
    def __init__(self, data_gen, meta_key=set(), filter: LZMAFilter=LZMAFilter.NONE):
        super().__init__(data_gen, meta_key=meta_key)
        self._filter = filter

    def compress_gen(self, max_length=-1):
        filters = lzma_filters[self._filter].copy()
        filters.append({'id': lzma.FILTER_LZMA2})
        compressor = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=filters)
        for data in self.data_gen(max_length=max_length):
            out = compressor.compress(data)
            if max_length and max_length > 0:
                out = [out[i:i+max_length] for i in range(0, len(out), max_length)]
                for o in out:
                    yield o
            else:
                yield out
        out = compressor.flush()
        if max_length and max_length > 0:
            out = [out[i:i+max_length] for i in range(0, len(out), max_length)]
            for o in out:
                yield o
        else:
            yield out

    @property
    def meta(self):
        return dict()


class LZMADecompressor(Decompressor):
    def __init__(self, compressed_data_gen, filter: LZMAFilter=LZMAFilter.NONE):
        super().__init__(compressed_data_gen)
        self._filter = filter

    def decompress_gen(self, max_length=-1):
        filters = lzma_filters[self._filter].copy()
        filters.append({'id': lzma.FILTER_LZMA2})
        decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
        for data in self.compressed_data_gen(max_length=max_length):
            yield decompressor.decompress(data, max_length=max_length)
            while not (decompressor.needs_input or decompressor.eof):
                yield decompressor.decompress(b'', max_length=max_length)


class DeflateCompressor(Compressor):
    def __init__(self, data_gen, meta_key=set()):
        super().__init__(data_gen, meta_key=meta_key)

    def compress_gen(self, max_length=-1):
        compressor = zlib.compressobj(wbits=-15)
        for data in self.data_gen(max_length=max_length):
            out = compressor.compress(data)
            if max_length and max_length > 0:
                out = [out[i:i+max_length] for i in range(0, len(out), max_length)]
                for o in out:
                    yield o
            else:
                yield out
        out = compressor.flush()
        if max_length and max_length > 0:
            out = [out[i:i+max_length] for i in range(0, len(out), max_length)]
            for o in out:
                yield o
        else:
            yield out

    @property
    def meta(self):
        return dict()


class DeflateDecompressor(Decompressor):
    def __init__(self, compressed_data_gen):
        super().__init__(compressed_data_gen)

    def decompress_gen(self, max_length=-1):
        if max_length < 0:
            max_length = 0
        decompressor = zlib.decompressobj(wbits=-15)
        for data in self.compressed_data_gen(max_length=max_length):
            yield decompressor.decompress(data, max_length=max_length)
            while decompressor.unconsumed_tail:
                yield decompressor.decompress(decompressor.unconsumed_tail, max_length=max_length)


class CompressedDataReader(CompressedDataGenerator):
    def __init__(self, reader: BlockDataReader, meta):
        super().__init__()
        self._reader = reader
        self._meta = meta

    def compress_gen(self, max_length=-1):
        reader = self._reader.copy()
        data = reader.read(max_length=max_length)
        while data != b'':
            yield data
            data = reader.read(max_length=max_length)

    @property
    def meta(self):
        return self._meta


class DecompressedDataGeneratorEncapsulator(DecompressedDataGenerator):
    def __init__(self, data_gen):
        super().__init__()
        self._data_gen = data_gen

    def decompress_gen(self, max_length=-1):
        return self._data_gen(max_length=max_length)


class DecompressedDataFileEncapsulator(DecompressedDataGenerator):
    def __init__(self, path: Union[str, PathLike]):
        super().__init__()
        self._path = path

    def decompress_gen(self, max_length=-1):
        with open(self._path, mode='rb') as f:
            data = f.read(max_length)
            while data != b'':
                yield data
                data = f.read(max_length)


class DecompressedDataBytesEncapsulator(DecompressedDataGenerator):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def decompress_gen(self, max_length=-1):
        if max_length and max_length > 0:
            for i in range(0, len(self._data), max_length):
                yield self._data[i:i+max_length]
        else:
            yield self._data
    

compressors = {
    CompressionType.NONE: NoneCompressor,
    CompressionType.LZMA: LZMACompressor,
    CompressionType.DEFLATE: DeflateCompressor,
} # type: Dict[CompressionType, Type[Compressor]]

decompressors = {
    CompressionType.NONE: NoneDecompressor,
    CompressionType.LZMA: LZMADecompressor,
    CompressionType.DEFLATE: DeflateDecompressor,
} # type: Dict[CompressionType, Type[Decompressor]]


class File(FileSystemObject):
    _compress_type: CompressionType
    _compressed_data_gen: Optional[CompressedDataGenerator]
    _data_gen: Optional[DecompressedDataGenerator]
    _read_meta: Dict[str, Any]

    def __init__(self, name: str, *args, **kwargs) -> None:
        self._compress_type = kwargs.pop('compression_type', CompressionType.DEFLATE) # deflate give reasonable size and speed
        self._compressed_data_gen = kwargs.pop('compressed_data_gen', None)
        self._data_gen = kwargs.pop('data_gen', None)
        self._read_meta = self._compressed_data_gen.meta if self._compressed_data_gen else {m: None for m in kwargs.pop('meta_key', tuple())}
        super().__init__(name, *args, **kwargs)

    @property
    def type(self):
        return FileSystemObjectType.FILE

    def copy(self):
        return File(
            self.name, 
            compression_type=self._compress_type, 
            compressed_data_gen=self._compressed_data_gen, 
            data_gen=self._data_gen, 
            meta_key=self._read_meta.keys()
        )

    @classmethod
    def instantiate_from_proxy(cls, proxy: 'FileSystemProxyObject') -> 'File':
        reader = proxy.reader().unlimited_reader
        print('load file', proxy.name)
        print('data at', reader._prev_tell-proxy.blockdata.base_offset)
        meta = proxy.meta
        return cls(proxy.name, compression_type=CompressionType(meta['compression_type']), compressed_data_gen=CompressedDataReader(reader, meta), parent=proxy.parent)

    def writer(self) -> FileSystemWriter:
        class Writer(FileSystemWriter):
            def __init__(self, file: 'File'):
                self._file = file

            def data(self, max_length=-1):
                return self._file.read_compressed(max_length=max_length)

            @property
            def require_offset(self) -> Dict[FileSystemObject, List[int]]:
                return dict()

        return Writer(self)

    def read(self, max_length=-1):
        if not self._data_gen:
            self._data_gen = decompressors[self._compress_type](self._compressed_data_gen.compress_gen)
        return self._data_gen.decompress_gen(max_length=max_length)

    def read_compressed(self, max_length=-1):
        if not self._compressed_data_gen:
            self._compressed_data_gen = compressors[self._compress_type](self._data_gen.decompress_gen, meta_key=self._read_meta.keys())
        return self._compressed_data_gen.compress_gen(max_length=max_length)

    @property
    def meta(self):
        if not self._compressed_data_gen:
            self._compressed_data_gen = compressors[self._compress_type](self._data_gen.decompress_gen, meta_key=self._read_meta.keys())
        meta = self._compressed_data_gen.meta
        compressiontype_meta = {'compression_type': int(self._compress_type)}
        if meta:
            return self._compressed_data_gen.meta.update(compressiontype_meta)
        else:
            return compressiontype_meta

    def replace_data(self, data_gen):
        self._compressed_data_gen = None
        self._data_gen = data_gen

    @classmethod
    def from_file_path(cls, path: Union[str, PathLike], *, name=None, compression_type=CompressionType.DEFLATE):
        return cls(name if name else Path(path).name, data_gen=DecompressedDataFileEncapsulator(path), compression_type=compression_type)

    @classmethod
    def from_bytes(cls, name, bytestr, *, compression_type=CompressionType.DEFLATE):
        return cls(name, data_gen=DecompressedDataBytesEncapsulator(bytestr), compression_type=compression_type)

    @classmethod
    def from_bytes_generator(cls, name, bytesgen, *, compression_type=CompressionType.DEFLATE):
        return cls(name, data_gen=DecompressedDataGeneratorEncapsulator(bytesgen), compression_type=compression_type)


filesystemobjects = {
    FileSystemObjectType.FOLDER: Folder,
    FileSystemObjectType.FILE: File,
} # type: Dict[FileSystemObjectType, Type[FileSystemObject]]


class TRF:
    _read_file: Optional[FileIO]
    _root: Folder

    def __init__(self, *args, **kwargs) -> None:
        self._read_file = kwargs.pop('file', None)
        self._root = kwargs.pop('root', Folder('.'))
        super().__init__(*args, **kwargs)

    def __getitem__(self, path: Union[str, PathLike]):
        return self._root[path]

    def write(self, file: Optional[FileIO]=None, max_length=-1):
        file = file if file and file is not self._read_file else TemporaryFile()
        self._read_file = file if file else self._read_file
        if not self._read_file:
            raise Exception('no file specified')

        file.write(b'TRF\x01')
        # encode reasonable max int size
        sz = sys.maxsize * 2 + 2
        i_sz = machine_int_size()
        while (sz >> 8) > 1:
            sz >>= 8
            file.write(b'\x00')
        else:
            _l_byte = 255
            while (sz >> 1) > 1:
                sz >>= 1
                _l_byte >>= 1
            file.write(bytes([_l_byte]))

        f_off = file.tell()
        curr_off = 0
        file.write(b'\x00'*i_sz); curr_off += i_sz # (will be) data length

        filesystem_offsets = dict()
        require_offsets = defaultdict(list)

        def write_object(fsobject: FileSystemObject):
            nonlocal curr_off
            filesystem_offsets[fsobject] = curr_off

            # HEADER
            file.write(b'\x00'*i_sz); curr_off += i_sz # block continuation offset from here (unsupported)
            length_off = curr_off
            file.write(b'\x00'*i_sz); curr_off += i_sz # (will be) block length
            # HEADER - NAME
            name = bytes(fsobject.name, encoding='utf-8')
            file.write(name)
            file.write(b'\xFF'); curr_off += len(name) + 1
            # HEADER - TYPE
            file.write(int(fsobject.type).to_bytes(1, 'little')); curr_off += 1
            # METADATA
            meta_length_off = curr_off
            file.write(b'\x00'*i_sz); curr_off += i_sz # (will be) meta length
            meta = cbor2.dumps(fsobject.meta)
            file.write(meta); curr_off += len(meta)
            # METADATA - LENGTH
            file.seek(f_off+meta_length_off, SEEK_SET)
            file.write((curr_off-meta_length_off-i_sz).to_bytes(i_sz, 'little'))
            file.seek(f_off+curr_off, SEEK_SET)
            # DATA
            data_off = curr_off
            writer = fsobject.writer()
            for data in writer.data(max_length=max_length):
                file.write(data); curr_off += len(data)
            # BLOCK LENGTH
            file.seek(f_off+length_off, SEEK_SET)
            file.write((curr_off-length_off-i_sz).to_bytes(i_sz, 'little'))
            file.seek(f_off+curr_off, SEEK_SET)
            # REQUIRED OFFSETS
            require_offsets.update({fso:[data_off+off for off in offs] for fso, offs in writer.require_offset.items()})
            # FOLDER SPECIFIC
            if isinstance(fsobject, Folder):
                for fso in fsobject.childs:
                    write_object(fso)

        write_object(self._root)
        print(filesystem_offsets)

        for f, req_offs in require_offsets.items():
            off = filesystem_offsets[f].to_bytes(i_sz, 'little')
            for req in req_offs:
                file.seek(f_off+req, SEEK_SET)
                file.write(off)

        file.seek(f_off, SEEK_SET)
        file.write((curr_off-i_sz).to_bytes(i_sz, 'little'))
        file.seek(f_off+curr_off, SEEK_SET)

        if file is not self._read_file:
            file.seek(0, SEEK_SET)
            shutil.copyfileobj(file, self._read_file)
            file.close()

    def add(self, fsobject: FileSystemObject):
        self._root.add(fsobject)

    @classmethod
    def read(cls, file: FileIO, max_length=-1) -> 'TRF':
        sig = read_exact_size(file, 4)
        if sig != b'TRF\x01':
            raise Exception('not a trf file')
        i_sz = 0
        while file.read(1) == b'\x00':
            i_sz += 1
        else:
            i_sz += 1
        f_off = file.tell()
        end_off = int.from_bytes(read_exact_size(file, i_sz), 'little')
        trf = TRF(file=file, root=FileSystemProxyObject(BlockData(file, f_off, i_sz, i_sz, _already_seek=True), max_length=max_length).load())

        file.seek(f_off+end_off, SEEK_SET)
        return trf
