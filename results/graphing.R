# globalEvalDf <- read.csv("global_eval.csv")
# malEvalDf <- read.csv("mal_eval.csv")
# malObjLogDf <- read.csv("mal_obj_log.csv")
# 
# plot(globalEvalDf$eval_success~globalEvalDf$t,xlab="Timestep",ylab="% Accuracy",main="Accuracy of Global Task over Time")
# lines(globalEvalDf$eval_success~globalEvalDf$t)
# 
# plot(malEvalDf$eval_loss~malEvalDf$t,xlab="Timestep",ylab="SSE",main="SSE Of Global and Malicious Tasks over Time",col="red")
# lines(globalEvalDf$eval_loss~globalEvalDf$t,col="blue")
# points(globalEvalDf$eval_loss~globalEvalDf$t,col="blue")
# lines(malEvalDf$eval_loss~malEvalDf$t,col="red")
# legend(1.7,600000,col = c("blue","red"),legend=c("Global","Malicious"),lty=1)
# 
# noContraDf <- read.csv("1_global_eval.csv")
# contraDf <- read.csv("contra.csv")
# plot(noContraDf$eval_loss~noContraDf$t,xlab="Timestep",ylab="SSE",main="SSE Of Global Task over Time",col="red")
# lines(contraDf$eval_loss~contraDf$t,col="blue")
# points(contraDf$eval_loss~contraDf$t,col="blue")
# lines(noContraDf$eval_loss~noContraDf$t,col="red")
# legend(2.5,50000,col = c("blue","red"),legend=c("No defense","CONTRA"),lty=1)

# pca_glob_df <- read.csv("pca_global_eval.csv")
# pca_mal_obj_df <- read.csv("pca_mal_obj.csv")
# nopca_glob_df <- read.csv("nopca_global_eval.csv")
# nopca_mal_obj_df <- read.csv("nopca_mal_obj.csv")
# 
# plot(pca_mal_obj_df$target_conf~pca_mal_obj_df$t,xlab="Timestep",ylab="",main="Global Task Accuracy and Malicious Confidence",col="red",ylim=c(0,1))
# lines(pca_mal_obj_df$target_conf~pca_mal_obj_df$t,col="red")
# points(pca_glob_df$eval_success/100~pca_glob_df$t,col="blue")
# lines(pca_glob_df$eval_success/100~pca_glob_df$t,col="blue")
# points(nopca_glob_df$eval_success/100~nopca_glob_df$t,col="green")
# lines(nopca_glob_df$eval_success/100~nopca_glob_df$t,col="green")
# points(nopca_mal_obj_df$target_conf~nopca_mal_obj_df$t,col="orange")
# lines(nopca_mal_obj_df$target_conf~nopca_mal_obj_df$t,col="orange")

# legend(20,0.6,col = c("blue","red","green","orange"),legend=c("PCA Accuracy","PCA Mal. Conf.","No Def. Acc.","No Def. Mal. Conf."),lty=1)



iid.df <- read.csv("iid.csv")
iid.df$acc_contra = iid.df$acc_contra/100
iid.df$acc_pca = iid.df$acc_pca/100
iid.df$acc_kernel = iid.df$acc_kernel/100
num_mal.df <- read.csv("num_mal.csv")
num_mal.df$acc_contra = num_mal.df$acc_contra/100
num_mal.df$acc_pca = num_mal.df$acc_pca/100
num_mal.df$acc_kernel = num_mal.df$acc_kernel/100
k.df <- read.csv("k.csv")
k.df$acc_contra = k.df$acc_contra/100
k.df$acc_pca = k.df$acc_pca/100
k.df$acc_kernel = k.df$acc_kernel/100

plot(iid.df$acc_contra~iid.df$iid,xlab="iid",ylab="Global Accuracy/Malicious Confidence",main="Global Accuracy and Malicious Confidence vs. iid",type="l",ylim=c(0,1),col="red")
lines(iid.df$acc_pca~iid.df$iid,col="blue")
lines(iid.df$acc_kernel~iid.df$iid,col="green")

lines(iid.df$mal_contra~iid.df$iid,col="red",lty=2)
lines(iid.df$mal_pca~iid.df$iid,col="blue",lty=2)
lines(iid.df$mal_kernel~iid.df$iid,col="green",lty=2)

legend(0.6,0.7,legend = c("Global Acc.","Mal. Conf.", "CONTRA","PCA","kPCA"),pch = c(NA, NA, 19,19,19),lty = c(1,2,NA,NA,NA),col = c("black","black","red","blue","green"))


plot(num_mal.df$acc_contra~num_mal.df$num_mal,xlab="Number of Malicious Clients",ylab="Global Accuracy/Malicious Confidence",main="Global Accuracy and Malicious Confidence vs. Number of Malicious Clients",type="l",ylim=c(0,1),col="red")
lines(num_mal.df$acc_pca~num_mal.df$num_mal,col="blue")
lines(num_mal.df$acc_kernel~num_mal.df$num_mal,col="green")

lines(num_mal.df$mal_contra~num_mal.df$num_mal,col="red",lty=2)
lines(num_mal.df$mal_pca~num_mal.df$num_mal,col="blue",lty=2)
lines(num_mal.df$mal_kernel~num_mal.df$num_mal,col="green",lty=2)

legend(9,0.7,legend = c("Global Acc.","Mal. Conf.", "CONTRA","PCA","kPCA"),pch = c(NA, NA, 19,19,19),lty = c(1,2,NA,NA,NA),col = c("black","black","red","blue","green"))


plot(k.df$acc_contra~k.df$k,xlab="Number of Clients",ylab="Global Accuracy/Malicious Confidence",main="Global Accuracy and Malicious Confidence vs. Number of Clients",type="l",ylim=c(0,1),col="red")
lines(k.df$acc_pca~k.df$k,col="blue")
lines(k.df$acc_kernel~k.df$k,col="green")

lines(k.df$mal_contra~k.df$k,col="red",lty=2)
lines(k.df$mal_pca~k.df$k,col="blue",lty=2)
lines(k.df$mal_kernel~k.df$k,col="green",lty=2)

legend(9,0.7,legend = c("Global Acc.","Mal. Conf.", "CONTRA","PCA","kPCA"),pch = c(NA, NA, 19,19,19),lty = c(1,2,NA,NA,NA),col = c("black","black","red","blue","green"))
