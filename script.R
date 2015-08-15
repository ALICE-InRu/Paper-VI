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
#FIGURE 3 \label{fig:opt:SDR}
plot.StepwiseSDR.wrtTrack(all.StepwiseOptimality,all.StepwiseExtremal,input$dimension,F,save,T)

source('opt.bw.R')
#FIGURE 5 \label{fig:case}
plot.BestWorst(input$problems,input$dimension,'OPT',save)

source('sdr.R')
#FIGURE 2 \label{fig:boxplot:SDR}
plot.SDR(SDR,'boxplot', save)
#FIGURE 4 \label{fig:boxplot:BDR}
plot.BDR(input$dimension,input$problems,'SPT','MWR',40,save)

source('pref.exhaustive.R'); source('pref.settings.R')
prefSummary=get.prefSummary(input$problems,input$dimension,'OPT','p',F)
paretoFront=get.paretoFront(prefSummary)
bestPrefModel=get.bestPrefModel(paretoFront)

#FIGURE 8 \label{fig:stepwise_vs_classification}
plot.exhaust.acc(prefSummary,save,bestPrefModel$Summary)

#FIGURE 9 \label{fig:CDR:scatter}
plot.exhaust.paretoFront(prefSummary,paretoFront,T,save)

for(problem in input$problems){
  #FIGURE 10 \label{fig:CDR:weights}
  print(plot.exhaust.paretoWeights(subset(paretoFront,Problem == problem),F,save))
}

#FIGURE 11 \label{fig:CDR:opt}
plot.exhaust.bestAcc(all.StepwiseOptimality,bestPrefModel,save)
x=dcast(subset(bestPrefModel$Stepwise,Accuracy=='Optimality'),Problem+Step~variable+Accuracy,value.var = 'value')
x=ddply(x,~Problem+Step,mutate,diff.acc=abs(Max.Acc.Opt_Optimality-Min.Rho_Optimality))
print(paste('Max oscillating',round(mean(x$diff.acc)*100),'%'))

#FIGURE 12 \label{fig:boxplot:CDR}
SDR=subset(SDR, (substr(Problem,1,1)=='j' & SDR=='MWR') |
             (substr(Problem,1,1)=='f' & SDR=='LWR'))
plot.exhaust.bestBoxplot(bestPrefModel,SDR,save,F)

#TABLE 3 \label{tbl:CDR:pareto}
cat(print(table.exhaust.paretoFront(paretoFront),
          include.rownames=FALSE, sanitize.text.function=function(x){x}),
    file = paste0(subdir,'../tables/PREF-',input$bias,'.tex'))

ks.readable <- function(ks,alpha=0.05){
  ks=unique(round(ks>1-alpha))
  ks=ks[rowSums(ks)>1,]
  if(is.matrix(ks)) ks=ks[,colSums(ks)>0]
  print(ks)
}

clc()
for(problem in input$problems){
  print(problem)
  ks=suppressWarnings(get.pareto.ks(paretoFront, problem, onlyPareto = F, SDR=NULL))
  if(!is.null(ks)){
    ks.readable(ks$Rho.train)
    ks.readable(ks$Acc)
  }
}

