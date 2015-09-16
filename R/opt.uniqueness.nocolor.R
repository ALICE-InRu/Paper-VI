get.StepwiseOptimality <- function(problems,dim,track='OPT'){

  get.StepwiseOptimality1 <- function(problem){
    fname=paste(paste0(DataDir,'Stepwise/optimality'),problem,dim,track,'csv',sep='.')

    if(file.exists(fname)){ split=read_csv(fname) } else {
      dat=get.files.TRDAT(problem, dim, track)
      if(is.null(dat)){ return(NULL) }
      dat$isOPT=dat$Rho==0

      split=ddply(dat,~Problem+Step+PID,summarise,
                  rnd=mean(isOPT),
                  unique=sum(isOPT),
                  .progress = "text")

      write.csv(split,file=fname,row.names=F,quote=F)
    }
    return(split)
  }

  split <- ldply(problems, get.StepwiseOptimality1)
  if(is.null(split)) { return(NULL) }

  split$Problem <- factorProblem(split)
  stats=ddply(split,~Problem+Step,summarise,
              rnd.mu=mean(rnd),
              rnd.Q1=quantile(rnd,.25),
              rnd.Q3=quantile(rnd,.75),
              unique.mu=mean(unique))

  if(max(stats$Step)<numericDimension(dim)){
    lastRow=stats[1,]
    lastRow$rnd.mu=1
    lastRow$rnd.Q1=1
    lastRow$rnd.Q3=1
    lastRow$unique.mu=1
    lastRow$Step=numericDimension(dim)
    stats=rbind(stats,lastRow)
  }

  return(list('Stats'=stats,'Raw'=split))
}


plot.stepwiseUniqueness <- function(StepwiseOptimality,dim,smooth,save=NA){
  if(is.null(StepwiseOptimality)) { return(NULL)}

  probs=levels(StepwiseOptimality$Stats$Problem)
  p=ggplot(StepwiseOptimality$Stats,aes(x=Step,y=unique.mu,linetype=Problem,fill=Problem))
  if(smooth){
    p=p+geom_smooth(aes(fill=Problem),alpha=0.1)+ggplotFill('Problem',length(probs))
  } else {
    p=p+geom_line()
  }

  p=p+ylab('Number of unique optimal dispatches')+
    axisStep(dim)+axisCompact

  if(!is.na(save)){
    fname=paste(paste(subdir,'stepwise',sep='/'),dim,'OPT','unique',extension,sep='.')
    if(save=='full')
      ggsave(filename=fname,plot=p, height=Height.full, width=Width, dpi=dpi, units=units)
    else if(save=='half')
      ggsave(filename=fname,plot=p, height=Height.half, width=Width, dpi=dpi, units=units)
  }

  return(p)
}

plot.stepwiseOptimality <- function(StepwiseOptimality,dim,simple,smooth,save=NA,asRND=F){
  if(is.null(StepwiseOptimality)) { return(NULL)}

  problems=levels(StepwiseOptimality$Stats$Problem)
  StepwiseOptimality$Stats$Problem <- factorProblem(StepwiseOptimality$Stats,F)
  StepwiseOptimality$Raw$Problem <- factorProblem(StepwiseOptimality$Raw,F)

  if(simple){
    p=ggplot(StepwiseOptimality$Stats,aes(x=Step,order=Problem))
    if(asRND)
      if(smooth){
        p=p+geom_smooth(data=StepwiseOptimality$Stats,aes(y=rnd.mu,linetype='RND',fill='RND'))
      } else {
        p=p+geom_line(aes(y=rnd.mu,linetype='RND'))
      }
    else
      p=p+geom_line(aes(y=rnd.mu))
  } else {
    p=ggplot(StepwiseOptimality$Stats,aes(x=Step,y=rnd.mu,linetype=Problem,fill=Problem))
    if(smooth){
      p=p+geom_smooth(aes(fill=Problem),alpha=0.1)+ggplotFill('Problem',length(problems))
    } else {
      p=p+geom_line()
    }
  }

  p=p+ylab('Probability of choosing optimal move')+
    axisStep(dim)+axisProbability

  if(!is.na(save)){
    fname=paste(paste(subdir,'stepwise',sep='/'),dim,'OPT',extension,sep='.')
    if(save=='full')
      ggsave(filename=fname,plot=p, height=Height.full, width=Width, dpi=dpi, units=units)
    else if(save=='half')
      ggsave(filename=fname,plot=p, height=Height.half, width=Width, dpi=dpi, units=units)
  }

  return(p)
}

