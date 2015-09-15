setwd('C:/Users/helga/alice/Code/R.shiny/')

source('global.R')
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
p <- p + scale_x_continuous('Step',limits=c(0,50),expand=c(0,0))
ggsave('../../JSP-Expert/figures/j_rnd/trdat_prob_moveIsOptimal_10x10_SDR_xihat.pdf',
       width = Width, height = Height.half, units = units, dpi = dpi)

source('opt.bw.R')
#FIGURE 5 \label{fig:case}
plot.BestWorst(input$problems,input$dimension,'OPT',save)

source('sdr.R')
#FIGURE 2 \label{fig:boxplot:SDR}
plot.SDR(SDR,'boxplot', save)
#FIGURE 4 \label{fig:boxplot:BDR}
plot.BDR(input$dimension,input$problem,'SPT','MWR',c(10,15,40),save,F)


source('pref.imitationLearning.R')
CDR.IL <- get.CDR.IL(input$problem,input$dimension)
CDR.IL <- subset(CDR.IL, Supervision == 'Unsupervised' | Iter==0)
plot.imitationLearning.boxplot(CDR.IL)
ggsave('../../JSP-Expert/figures/j_rnd/DAGGER_10x10.pdf',
       width = Width, height = Height.half, units = units, dpi = dpi)

stats.imitationLearning(CDR.IL)
plot.imitationLearning.weights(input$problem,input$dimension)
