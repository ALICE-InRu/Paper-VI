if(Sys.info()['user']=="hei2"){  setwd('~/alice/Code/R')  
} else if(Sys.info()['user']=="helga"){ setwd('C:/Users/Helga/alice/Code/R') }

rm(list=ls(all=TRUE))
source('myFigures.R')
extension='pdf';mainPalette='Set1';subdir='../../Papers/JOH.Features/figures/'

source('getfiles.R')
# -------------------------------------------------------------------
OPT=getOPTs()
# ----- Box plots for SDRs -----------------------------------------
SDR=getfiles('../SDR/','f.rnd|j.rnd')
SDR <- subset(SDR, Name %in% OPT$Name)
SDR <- join(SDR,OPT[,c('Name','Optimum')],by='Name',type='inner')

SDR=subset(SDR,Set=='train' & Dimension=='10x10')
SDR=subset(SDR,Problem != 'j.rnd, J1' & Problem != 'j.rnd, M1' & Problem!='f.rndn')

SDR=as.data.table(SDR)
setkey(SDR,'Dimension', 'Problem','Set')

# ------------------------------------------------------------------
source('difficultywrtSDR.R'); 
p=boxplotSDR(SDR)
fname=paste(subdir,'boxplotRho.SDR.10x10.pdf',sep='')
ggsave(fname,width=WidthMM,height=HeightMM.half,dpi=dpi,units=units)

# ------------------------------------------------------------------
source('inspectBDR.R')
p=checkBDR(OPT,SDR,'SPT','MWR',40)
p = p + ggplotFill('Dispatching rule',3,c('SPT (first 40%), MWR (last 60%)','Most Work Remaining','Shortest Processing Time'))
fname=paste(subdir,'boxplotRho.BDR.10x10.pdf',sep='')
ggsave(fname,width=WidthMM,height=HeightMM.half,dpi=dpi,units=units)

# ------------------------------------------------------------------
source('optimalityOfDispatches.R')
inspectOptimalityFromMatlab()

# ------------------------------------------------------------------
source('difficultywrtFeatures.R');
TRDAT.10x10=getfilesTraining('10x10',Global=F)
features.evolution(TRDAT.10x10)