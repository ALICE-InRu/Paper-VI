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
ggsave('../../JSP-Expert/figures/j_rnd/trdat_prob_moveIsOptimal_10x10_SDR_xi.pdf',
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
CDR.IL <- do.call(rbind, lapply(c('equal','adjdbl2nd'), function(bias) {
  get.CDR.IL(input$problem,input$dimension,bias)}))
CDR.IL <- subset(CDR.IL, (Supervision == 'Unsupervised' & Iter <=3) | Iter==0 )
CDR.IL$Set=interaction(CDR.IL$Problem,CDR.IL$Dimension,CDR.IL$Set,sep=', ')

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
plot.imitationLearning.boxplot(CDR.DA.EXT)+facet_grid(Type~Set)+
  scale_linetype('New instances')+xlab(expression('iteration,' *~i))
ggsave('../../JSP-Expert/figures/j_rnd/boxplot_active_10x10.pdf',
       width = Width, height = Height.half, units = units, dpi = dpi)

CDR.IL$Type = 'Imitation Learning'
CDR.IL <- subset(CDR.IL, (Supervision=='Unsupervised' & Extended==T) |
                   (Supervision!='Unsupervised' & Extended==F))

stats.imitationLearning(CDR.IL)

stat=ddply(CDR.IL,~Track+Extended+Iter+Bias+Set,function(x) c(nrow(x),summary(x$Rho)))
print(xtable(stat),include.rownames=F)

MWR <- subset(dataset.SDR,SDR=='MWR' & Problem == input$problem & Dimension==input$dimension);
MWR$Set=interaction(MWR$Problem,MWR$Dimension,MWR$Set,sep=', ')
CDR.IL$Supervision <- factor(CDR.IL$Supervision,levels=c('Fixed','Unsupervised'),labels=c('Passive Imitation Learning','Active Imitation Learning'))

plot.imitationLearning.boxplot(CDR.IL,MWR)
ggsave('../../JSP-Expert/figures/j_rnd/boxplot_summary_10x10.png',
       width = Width, height = Height.half, units = units, dpi = dpi)
# to get epsilon right: gm convert boxplot_summary_10x10.png boxplot_summary_10x10.pdf


plot.imitationLearning.weights(input$problem,input$dimension)
