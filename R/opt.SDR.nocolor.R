get.StepwiseExtremal <- function(problems,dim){

  get.StepwiseExtremal1 <- function(problem){

    fname=paste(paste0(DataDir,'Stepwise/extremal'),problem,dim,'csv',sep='.')

    if(file.exists(fname)){ split=read_csv(fname)
    } else {
      trdatL=get.files.TRDAT(problem, dim, 'OPT', Global = F)
      if(is.null(trdatL)) { return(NULL)}
      trdatG=get.files.TRDAT(problem, dim, 'OPT', Global = T)
      trdat=join(trdatL,trdatG,by=colnames(trdatG)[colnames(trdatG) %in% colnames(trdatL)])

      trdat$isOPT=trdat$Rho==0
      mdat=melt(trdat, variable.name = 'Feature',
                measure.vars = colnames(trdat)[grep('phi',colnames(trdat))])

      split=NULL
      for(phi in levels(mdat$Feature)){
        # cannot apply ddply on 10x10 all at once (out of memory)
        tmp=ddply(subset(mdat,Feature==phi),~Problem+Step+PID+Feature,summarise,
                  max=mean(isOPT[value==max(value)]),
                  min=mean(isOPT[value==min(value)]),
                  .progress = "text")

        split=rbind(split,tmp)
      }
      write.csv(split,file=fname,row.names=F,quote=F)
    }
    return(split)
  }

  split <- ldply(problems, get.StepwiseExtremal1)
  if(is.null(split)) { return(NULL) }

  split=subset(split,Feature != 'phi.step' & Feature != 'phi.wrmTotal') # min == max

  split$Problem <- factorProblem(split)
  split$Feature <- factorFeature(split$Feature)

  split=melt(split,measure.vars = c('min','max'),variable.name = 'Extremal')
  stats=ddply(split,~Problem+Step+Feature+Extremal,summarise,Extremal.mu=mean(value))

  return(list('Stats'=stats,'Raw'=split))
}

plot.StepwiseSDR.wrtTrack <- function(StepwiseOptimality,StepwiseExtremal,
                                      dim,smooth,save=NA,onlyWrtOPT=F,onlyWrtSDR=F){
  if(is.null(StepwiseOptimality)|is.null(StepwiseExtremal)) {return(NULL)}

  problems <- levels(StepwiseOptimality$Stats$Problem)

  plot.StepwiseSDR.wrtOPT <- function(){

    SDR=subset(StepwiseExtremal$Raw, Feature=='proc' | Feature=='jobWrm')
    SDR$SDR[SDR$Feature=='proc' & SDR$Extremal=='min']='SPT'
    SDR$SDR[SDR$Feature=='proc' & SDR$Extremal=='max']='LPT'
    SDR$SDR[SDR$Feature=='jobWrm' & SDR$Extremal=='min']='LWR'
    SDR$SDR[SDR$Feature=='jobWrm' & SDR$Extremal=='max']='MWR'
    SDR$SDR=factorSDR(SDR$SDR)
    SDR$Problem <- factorProblem(SDR,F)

    p=plot.stepwiseOptimality(StepwiseOptimality,dim,T,smooth,asRND = T) # random guessing

    if(smooth){
      p=p+geom_smooth(data=SDR,aes(y=value,linetype=SDR,fill=SDR))+
        ggplotFill('SDR',length(sdrs),values=sdrs)
    } else {
      stat=ddply(SDR,~Problem+Step+SDR,summarise,mu=mean(value))
      p=p+geom_line(data=stat,aes(y=mu,linetype=SDR))
    }

    return(p)
  }

  if(!onlyWrtSDR){
    p=plot.StepwiseSDR.wrtOPT()
  } else {
    p <- ggplot(NULL,aes(x=Step)) + axisStep(dim) + axisCompact
  }
  p = p + facet_wrap(~Problem,ncol=3)

  if(!onlyWrtOPT){
    SDR <- do.call(rbind, lapply(sdrs, function(sdr) { data.frame(Track = sdr, get.StepwiseOptimality(problems,dim,sdr)$Stats)} ))
  } else {SDR=NULL}

  if(!is.null(SDR)){
    SDR<- factorTrack(SDR)
    SDR$Problem <- factorProblem(SDR,F)
    if(onlyWrtSDR){ SDR$rnd.mu = log(SDR$rnd.mu) }
    p=p+geom_line(data=SDR,aes(y=rnd.mu,linetype=Track))
  }

  if(onlyWrtOPT|onlyWrtSDR) {
    if(onlyWrtOPT)
      p <- p+ylab(expression(xi[SDR]^'*'))+scale_linetype('Track')
    else
      p <- p+ylab(expression(log(xi[SDR])))

  } else {
    p <- p + ylab('Probability of SDR being optimal')
    p=p+scale_linetype_manual('Accuracy', values=c(1,2),
                              labels=c(expression(xi^'*'),expression(xi)))
  }

  if(!is.na(save)){
    fname=ifelse(length(problems)>1,
                 paste(paste(subdir,'stepwise',sep='/'),dim,'SDR',extension,sep='.'),
                 paste(paste(subdir,problems,'stepwise',sep='/'),dim,'SDR',extension,sep='.'))

    if(save=='full')
      ggsave(filename=fname,plot=p, height=Height.full, width=Width, dpi=dpi, units=units)
    else if(save=='half')
      ggsave(filename=fname,plot=p, height=Height.half, width=Width, dpi=dpi, units=units)
  }

  return(p)
}
