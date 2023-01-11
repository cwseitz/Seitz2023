from tiler import Tiler
from correct import Basic
from detect import Detector
from gptie import GPTIE
#from nucleus_model import NucleusModel
#from cell_model import CellBodyModel
from segment import CellSegmenter
from pathlib import Path

class Pipeline:
    def __init__(self,config,prefix):
        self.config = config
        self.datapath = config['datapath']
        self.analpath = config['analpath']
        self.cmodelpath = config['cmodelpath']
        self.nmodelpath = config['nmodelpath']
        self.cell_filters = config['cell_filters']
        self.nucleus_filters = config['nucleus_filters']
        self.ch1_thresh = config['ch1_thresh']
        self.ch2_thresh = config['ch2_thresh']
        self.p0 = config['p0']
        self.prefix = prefix
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
    def sequential(self):
        self.tile()
        self.basic_correct()
        self.apply_nucleus_model()
        self.apply_cell_model()
        self.detect_spots()
        self.segment_cells()
        self.spot_counts()
    def tile(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_ch0.tif')
        if not file.exists():
            print('Tiling raw data...')
            tiler = Tiler(self.datapath,self.analpath,self.prefix)
            tiler.tile()
        else:
            print('Tiled data exists. Skipping')
    def basic_correct(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_mxtiled_corrected_ch0.tif')
        if not file.exists():
            print('Applying basic correction...')
            basic = Basic(self.analpath,self.prefix)
            basic.correct()
        else:
            print('Corrected files exists. Skipping')
    def apply_nucleus_model(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_ch0_softmax.npz')
        if not file.exists():
            print('Applying nucleus model...')
            nmodel = NucleusModel(self.nmodelpath,self.analpath,self.prefix,self.nucleus_filters)
            nmodel.apply()
        else:
            print('Nucleus softmax files exist. Skipping')
    def apply_cell_model(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_ch1_softmax.npz')
        if not file.exists():
            print('Applying cell model...')
            cmodel = CellBodyModel(self.cmodelpath,self.analpath,self.prefix,self.cell_filters)
            cmodel.apply()
        else:
            print('Cell softmax files exist. Skipping')
    def detect_spots(self,plot=False):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_ch1_spots.csv')
        if not file.exists():
            print('Running spot detection...')
            detector = Detector(self.datapath,self.analpath,self.prefix,self.ch1_thresh,self.ch2_thresh)
            detector.detect(plot=plot)
        else:
            print('Spot files exist. Skipping')
    def segment_cells(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_ch1_mask.tif')
        if not file.exists():
            print('Running cell segmentation...')
            cellsegment = CellSegmenter(self.datapath,self.analpath,self.prefix,self.cell_filters,self.p0)
            cellsegment.segment()
        else:
            print('Mask files exist. Skipping')
    def volume_contrast(self):
        file = Path(self.analpath+self.prefix+'/'+self.prefix+'_ch3_gptie.tif')
        if not file.exists():
            print('Running GPTIE Model...')
            gptie = GPTIE(self.datapath,self.analpath,self.prefix)
            gptie.apply()
        else:
            print('Mask files exist. Skipping')      
  

