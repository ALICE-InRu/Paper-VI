setwd('C:/Users/helga/alice/Code/R.shiny/')

source('global.R')
subdir='../../JSP-Expert/figures/'
save='half';extension='pdf'
input=list(dimension='10x10',problem='j.rnd',problems=c('j.rnd','j.rndn','f.rnd'))
SDR=subset(dataset.SDR,Problem %in% input$problems & Dimension %in% input$dimension)
input$bias='equal'
input$timedependent=F
input$smooth=F

source('opt.uniqueness.R');
all.StepwiseOptimality=get.StepwiseOptimality(input$problems,input$dimension,'OPT')
#FIGURE 2
plot.stepwiseUniqueness(all.StepwiseOptimality,input$dimension,input$smooth,save)
#FIGURE 3
plot.stepwiseOptimality(all.StepwiseOptimality,input$dimension,F,input$smooth,save)

source('opt.SDR.R')
all.StepwiseExtremal=get.StepwiseExtremal(input$problems,input$dimension)
plot.StepwiseSDR.wrtTrack(all.StepwiseOptimality,all.StepwiseExtremal,input$dimension,F,'full')

source('opt.bw.R')
#FIGURE 4
plot.BestWorst(input$problems,input$dimension,'OPT',save)

source('opt.SDR.R')
StepwiseExtremal=get.StepwiseExtremal(input$problems,input$dimension)
#FIGURE 5
plot.StepwiseSDR.wrtTrack(all.StepwiseOptimality,StepwiseExtremal,input$dimension,F)

source('sdr.R')
#FIGURE 6
plot.SDR(SDR,'boxplot', save)
#FIGURE 7
plot.BDR(input$dimension,input$problems,'SPT','MWR',40,save)

source('pref.exhaustive.R'); source('pref.settings.R')
prefSummary=get.prefSummary(input$problems,input$dimension,'OPT','p',F,input$bias)
paretoFront=get.paretoFront(prefSummary)
bestPrefModel=get.bestPrefModel(paretoFront)

#FIGURE 8
plot.exhaust.acc(prefSummary,save)

#FIGURE 9
plot.exhaust.paretoFront(prefSummary,paretoFront,T,save)

#FIGURE 10
plot.exhaust.bestAcc(all.StepwiseOptimality,bestPrefModel,save)
x=dcast(subset(bestPrefModel$Stepwise,Accuracy=='Optimality'),Problem+Step~variable+Accuracy,value.var = 'value')
x=ddply(x,~Problem+Step,mutate,diff.acc=abs(Max.Acc.Opt_Optimality-Min.Rho_Optimality))
print(paste('Max oscillationg',round(mean(x$diff.acc)*100),'%'))

#FIGURE 11
SDR=subset(SDR, (substr(Problem,1,1)=='j' & SDR=='MWR') |
             (substr(Problem,1,1)=='f' & SDR=='LWR'))
plot.exhaust.bestBoxplot(bestPrefModel,SDR,save,F)

for(problem in input$problems){
  #FIGURE 12
  print(plot.exhaust.paretoWeights(subset(paretoFront,Problem == problem),F,save))
}

#TABLE 3
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

