"""Generate a deterministic artificial OME-TIFF fixture; no biological data."""
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, shift
import tifffile


def create_demo(path):
    path=Path(path)
    if path.exists():
        raise ValueError('Demo output exists; choose a new filename.')
    rng=np.random.default_rng(1234)
    T,C,Z,Y,X=8,2,3,96,160
    data=rng.normal(20,1.5,size=(T,C,Z,Y,X))
    yy,xx=np.indices((Y,X))
    shaft=70*np.exp(-((yy-48)/3)**2)*((xx>15)&(xx<145))
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                data[t,c,z]+=shaft*np.exp(-((z-1)/.8)**2)
        for x in range(26,140,12):
            blob=np.zeros((Z,Y,X)); blob[1,48,x]=800
            blob=gaussian_filter(blob,sigma=(.5,1.,1.))
            for c in range(C):
                strength=1.0 if t>=1+c else .04
                data[t,c]+=strength*blob
        data[t]=shift(data[t],(0,0,t*.12,t*.08),order=1,mode='nearest')
    data=np.clip(data,0,65535).astype(np.uint16)
    path.parent.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(path,data,ome=True,photometric='minisblack',metadata={
        'axes':'TCZYX','PhysicalSizeX':.4,'PhysicalSizeY':.4,'PhysicalSizeZ':.8,
        'PhysicalSizeXUnit':'µm','PhysicalSizeYUnit':'µm','PhysicalSizeZUnit':'µm',
        'TimeIncrement':2.,'TimeIncrementUnit':'s','Channel':{'Name':['A','B']}})
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    create_demo(args.output)
    print('Wrote artificial data; this fixture does not establish biological accuracy.')
