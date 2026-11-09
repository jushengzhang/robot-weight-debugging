#!/usr/bin/env python3
"""Build a float32 .npy override from a compatible flat safetensors checkpoint.
Does not run a model. Requires numpy. Never modifies the input checkpoint.
"""
import argparse,json,os,struct
from pathlib import Path
import numpy as np

def table(path,tensor):
 with path.open('rb') as f:
  raw=f.read(8)
  if len(raw)!=8:raise ValueError('Invalid safetensors header')
  size=struct.unpack('<Q',raw)[0]
  if size>100_000_000:raise ValueError('Unexpected safetensors header size')
  meta=json.loads(f.read(size))
 if tensor not in meta:raise ValueError('Required tensor is absent')
 entry=meta[tensor];dtype=entry['dtype']
 dt={'BF16':'<u2','F32':'<f4','F16':'<f2'}.get(dtype)
 if dt is None:raise ValueError('Unsupported source dtype: '+dtype)
 source=np.memmap(path,mode='r',dtype=dt,offset=8+size+entry['data_offsets'][0],shape=tuple(entry['shape']))
 def convert(x):
  if dtype=='BF16':return (np.asarray(x,dtype=np.uint32)<<16).view(np.float32)
  return np.asarray(x,dtype=np.float32)
 return source,convert

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('checkpoint',type=Path,help='Original pi05_libero_flat.safetensors')
 p.add_argument('--payload',type=Path,default=Path(__file__).with_name('pi05-lang-c-sparse-rows.npz'))
 p.add_argument('--recipe',type=Path,default=Path(__file__).with_name('embedding-recipe.json'))
 p.add_argument('--output',type=Path,help='New .npy override path; existing files are refused')
 p.add_argument('--dry-run',action='store_true',help='Check original rows without writing a full tensor')
 a=p.parse_args();recipe=json.loads(a.recipe.read_text());data=np.load(a.payload,allow_pickle=False)
 ids=data['row_ids'].astype(np.int64);original=data['original_rows'];edited=data['edited_rows']
 source,convert=table(a.checkpoint,recipe['tensor'])
 if list(source.shape)!=recipe['shape']:raise ValueError('Checkpoint tensor shape mismatch')
 if ids.ndim!=1 or len(np.unique(ids))!=len(ids):raise ValueError('Invalid or duplicate row IDs')
 if original.shape!=edited.shape or original.shape!=(len(ids),source.shape[1]):raise ValueError('Payload shape mismatch')
 if np.any(ids<0) or np.any(ids>=source.shape[0]):raise ValueError('Row IDs out of bounds')
 if ids.tolist()!=[r['row'] for r in recipe['rows']]:raise ValueError('Payload/recipe row IDs disagree')
 if not np.array_equal(convert(source[ids]),original):raise ValueError('Original row values mismatch; use the evaluated base checkpoint')
 if not np.isfinite(edited).all():raise ValueError('Nonfinite edited row values')
 print(f'Verified {len(ids)} original rows; output shape {tuple(source.shape)}')
 if a.dry_run:return
 if a.output is None:p.error('--output is required unless --dry-run')
 output=a.output.resolve()
 if output.suffix!='.npy':raise ValueError('Output must end in .npy')
 if output==a.checkpoint.resolve() or output.exists():raise ValueError('Refusing to replace an existing file')
 output.parent.mkdir(parents=True,exist_ok=True)
 temporary=output.with_name(output.name+'.partial')
 fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
 try:
  dest=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=source.shape)
  for start in range(0,len(source),4096):dest[start:start+4096]=convert(source[start:start+4096])
  dest[ids]=edited;dest.flush();del dest
  check=np.load(temporary,mmap_mode='r')
  if not np.array_equal(check[ids],edited):raise ValueError('Written rows failed verification')
  del check
  # Hard-link creation refuses a destination introduced concurrently.
  os.link(temporary,output);temporary.unlink()
 except BaseException:
  if temporary.exists():temporary.unlink()
  raise
 print(f'Wrote {output}; install as a replacement for {recipe["tensor"]}')
if __name__=='__main__':main()
