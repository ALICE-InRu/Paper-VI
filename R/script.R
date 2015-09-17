setwd('C:/Users/helga/alice/Code/R.shiny/')

source('global.R')
colorPalette='Greys';
subdir='../../JSP-Expert/figures/'
save='half';extension='pdf'
input=list(dimension='10x10',problem='j.rnd',problems=c('j.rnd','j.rndn','f.rnd'))
SDR=subset(dataset.SDR,Problem %in% input$problems & Dimension %in% input$dimension)
input$timedependent=F
input$smooth=F

source('opt.uniqueness.R');
all.StepwiseOptimality=get.StepwiseOptimality(input$problems,input$dimension,'OPT')
#FIGURE NA \label{fig:opt:unique}
#plot.stepwiseUniqueness(all.StepwiseOptimality,input$dimension,input$smooth,save)
#FIGURE NA \label{fig:opt}
#plot.stepwiseOptimality(all.StepwiseOptimality,input$dimension,F,input$smooth,save)

source('opt.SDR.R')
all.StepwiseExtremal=get.StepwiseExtremal(input$problems,input$dimension)
#FIGURE 3 \label{fig:opt:SDR:xi}
p=plot.StepwiseSDR.wrtTrack(all.StepwiseOptimality,all.StepwiseExtremal,input$dimension,F,save,T)
#FIGURE 6 \label{fig:opt:SDR:xihat:jrnd}
jrnd.StepwiseOptimality=get.StepwiseOptimality(input$problem,input$dimension,'OPT')
jrnd.StepwiseExtremal=get.StepwiseExtremal(input$problem,input$dimension)
p=plot.StepwiseSDR.wrtTrack(jrnd.StepwiseOptimality,jrnd.StepwiseExtremal,input$dimension,F,NA,F,T)
p <- p + scale_x_continuous('Step',limits=c(0,30),expand=c(0,0))
ggsave('../../JSP-Expert/figures/j_rnd/trdat_prob_moveIsOptimal_10x10_SDR_xihat.pdf',
       width = Width, height = Height.half, units = units, dpi = dpi)

source('opt.bw.R')
#FIGURE 5 \label{fig:case}
plot.BestWorst(input$problems,input$dimension,'OPT',save)

source('sdr.R')
#FIGURE 2 \label{fig:boxplot:SDR}
plot.SDR(SDR,'boxplot', save)
#FIGURE 4 \label{fig:boxplot:BDR}
plot.BDR(input$dimension,input$problem,'SPT','MWR',c(10,15,20,40),NA,F)


source('pref.imitationLearning.R')
CDR.IL <- get.CDR.IL(input$problem,input$dimension)
CDR.IL <- subset(CDR.IL, (Supervision == 'Unsupervised' & Iter <=3) | Iter==0 )

CDR.OPT <- subset(CDR.IL, Iter==0)
CDR.OPT$Type <- 'Passive Imitation Learning'
plot.imitationLearning.boxplot(CDR.OPT)+guides(colour=FALSE)+facet_grid(Type~Set)
ggsave('../../JSP-Expert/figures/j_rnd/boxplot_passive_10x10.png',
       width = Width, height = Height.half, units = units, dpi = dpi)
# to get epsilon right: gm convert boxplot_passive_10x10.png boxplot_passive_10x10.pdf

CDR.DA.EXT <- subset(CDR.IL, (Supervision == 'Unsupervised') | (Extended==0 & Track=='OPT'))
CDR.DA.EXT$Type <- 'Active Imitation Learning'
CDR.DA.EXT$Supervision='Fixed'
levels(CDR.DA.EXT$Track)[1]='DA0'
plot.imitationLearning.boxplot(CDR.DA.EXT)+guides(colour=FALSE)+facet_grid(Type~Set)+
  scale_linetype('New problem instances')+xlab(expression('iteration,' *~T))
ggsave('../../JSP-Expert/figures/j_rnd/boxplot_active_10x10.pdf',
       width = Width, height = Height.half, units = units, dpi = dpi)

CDR.IL$Type = 'Imitation Learning'
CDR.IL <- subset(CDR.IL, (Supervision=='Unsupervised' & Extended==T) |
                   (Supervision!='Unsupervised' & Extended==F))
MWR <- subset(dataset.SDR,SDR=='MWR' & Problem == input$problem & Dimension==input$dimension);
MWR$Supervision='Unsupervised'; MWR$Iter=0; MWR$Track='MWR'; MWR$CDR=MWR$SDR; MWR$NrFeat=1; MWR$Model=1; MWR$Extended=F; MWR$Bias=NA; MWR$Rank=NA; MWR$SDR=NULL; MWR$Type='SDR'
p=plot.imitationLearning.boxplot(rbind(CDR.IL,MWR))
p=p+ggplotColor('IL',2,values=c('Fixed','Unsupervised'),labels=c('Passive','Active'))
p=p + facet_grid(Set~Type, scales='free',space = 'free')
ggsave('../../JSP-Expert/figures/j_rnd/boxplot_summary_10x10.png',
       width = Width, height = Height.half, units = units, dpi = dpi)
# to get epsilon right: gm convert boxplot_summary_10x10.png boxplot_summary_10x10.pdf

stats.imitationLearning(CDR.IL)
plot.imitationLearning.weights(input$problem,input$dimension)
