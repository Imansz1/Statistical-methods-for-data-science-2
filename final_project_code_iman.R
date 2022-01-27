
# Load Libraries
library(ggplot2)
library(gridExtra)
library(caret)
library(performanceEstimation)
library(scales)
library(reshape2)
library(rjags)
library(jagsUI)
library(coda)

# Set path
setwd("C:/Users/Iman/Desktop/sdsp")

# set random seed
seed <- 0

options(scipen=999)

# Read data
data <- read.csv("diabetes.csv")


####EDA####
summary(data)
var(data)
cor(data)

data$Outcome_factor <- as.factor(data$Outcome)

p1 <- ggplot(data, aes(x=Outcome_factor, y=Pregnancies, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)
  

p2 <- ggplot(data, aes(x=Outcome_factor, y=Glucose, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p3 <- ggplot(data, aes(x=Outcome_factor, y=BloodPressure, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p4 <- ggplot(data, aes(x=Outcome_factor, y=SkinThickness, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p5 <- ggplot(data, aes(x=Outcome_factor, y=Insulin, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p6 <- ggplot(data, aes(x=Outcome_factor, y=BMI, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p7 <- ggplot(data, aes(x=Outcome_factor, y=DiabetesPedigreeFunction, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)

p8 <- ggplot(data, aes(x=Outcome_factor, y=Age, fill=Outcome_factor)) +
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1)


grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol=2)


# Calculate correlation matrix
cormat <- round(cor(data[,c("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age")]),2)
cormat


# Response variable
blank_theme <- theme_minimal()+
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.border = element_blank(),
    panel.grid=element_blank(),
    axis.ticks = element_blank(),
    plot.title=element_text(size=14, face="bold")
  )

pie_data <- as.data.frame(table(data$Outcome))
pie_data$perc <- round(pie_data$Freq / sum(pie_data$Freq),2)*100
bp<- ggplot(pie_data, aes(x="", y=Freq, fill=Var1))+
  geom_bar(width = 1, stat = "identity")
pie <- bp + coord_polar("y", start=0) + theme_minimal()+blank_theme+ theme(axis.text.x=element_blank()) +
  geom_text(aes(x=1.1, label = paste(perc,"%",sep="")), size=8, position = position_stack(vjust = .5)) + theme(legend.position="right") +
  theme(
    text = element_text(size=20),
    plot.title = element_text(size=30)
  )
pie 

#### Fit a model with Bayesian model logit version####
# Define model
#Identify filepath of model file
modfile_logit <- tempfile()

#Write model to file
writeLines("
model{

  # Likelihood
  for (i in 1:N){
    y[i] ~ dbern(pi[i])
    
    logit(pi[i]) <- beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i] + beta_4*x4[i] + beta_5*x5[i] + beta_6*x6[i] + beta_7*x7[i] + beta_8*x8[i]
  }
  
  # Define prior for intercept
  beta_0 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta1 (Pregnancies)
  beta_1 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta2 (Glucose)
  beta_2 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta3 (BloodPressure)
  beta_3 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta4 (SkinThickness)
  beta_4 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta5 (Insulin)
  beta_5 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta6 (BMI)
  beta_6 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta7 (DiabetesPedigreeFunction)
  beta_7~ dnorm(0, 1.0E-6)
  
  # Define prior for beta8 (Age)
  beta_8 ~ dnorm(0, 1.0E-6)

}
", con = modfile_logit)


# Preparing data for JAGS
# Set number of individuals
N <- dim(data)[1]

# Response variable
y <- as.vector(as.numeric(as.character(data$Outcome))) # Target variable, risks heart attack?

data_jags <- list("y" = y, "N" = N, "x1" = data$Pregnancies
                  , "x2" = data$Glucose
                  , "x3" = data$BloodPressure
                  , "x4" = data$SkinThickness
                  , "x5" = data$Insulin
                  , "x6" = data$BMI
                  , "x7" = data$DiabetesPedigreeFunction
                  , "x8" = data$Age
                  )

# Defining parameters of interest
parameters <- c("beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5", "beta_6", "beta_7", "beta_8")

# Fit model with JAGS
set.seed(seed)
mod_fit_logit <- jags(data = data_jags,                                               
                         model.file = modfile_logit,                                             
                         parameters.to.save = parameters,                                
                         n.chains = 1, n.iter = 10000, n.burnin = 1000)



# Get AIC of the model
mod_fit_logit$DIC

# Get summary of the model
mod_fit_logit$summary


####### Defining second logit model by dropping non-significant covariates #######                                  

# Define model
#Identify filepath of model file
modfile_logit2 <- tempfile()

#Write model to file
writeLines("
model{

  # Likelihood
  for (i in 1:N){
    y[i] ~ dbern(pi[i])
    
    logit(pi[i]) <- beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i] + beta_4*x4[i] + beta_5*x5[i] 
  }
  
  # Define prior for intercept
  beta_0 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta1 (Pregnancies)
  beta_1 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta2 (Glucose)
  beta_2 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta3 (BloodPressure)
  beta_3 ~ dnorm(0, 1.0E-6)

  # Define prior for beta4 (BMI)
  beta_4 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta5 (DiabetesPedigreeFunction)
  beta_5~ dnorm(0, 1.0E-6)
  

}
", con = modfile_logit2)


# Preparing data for JAGS
# Set number of individuals
N <- dim(data)[1]

# Response variable
y <- as.vector(as.numeric(as.character(data$Outcome)))

data_jags2 <- list("y" = y, "N" = N, "x1" = data$Pregnancies
                  , "x2" = data$Glucose
                  , "x3" = data$BloodPressure
                  , "x4" = data$BMI
                  , "x5" = data$DiabetesPedigreeFunction
)

# Defining parameters of interest
parameters2 <- c("beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5")

# Fit model with JAGS
set.seed(seed)
mod_fit_logit2 <- jags(data = data_jags2,                                               
                      model.file = modfile_logit2,                                             
                      parameters.to.save = parameters2,                                
                      n.chains = 1, n.iter = 10000, n.burnin = 1000)

# Get AIC of the model
mod_fit_logit2$DIC

# Get summary of the model
mod_fit_logit2$summary

g1 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_0), aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_0, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[0])))


g2 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_0), aes(x=mod_fit_logit2$sims.list$beta_0))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[0])))


bacf <- acf(mod_fit_logit2$sims.list$beta_0, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g3 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+labs(title=expression(paste("Autocorrelogram of ",beta[0])))

g4 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_1)
             , aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_1, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[1])))

g5 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_1)
             , aes(x=mod_fit_logit2$sims.list$beta_1))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[1])))

bacf <- acf(mod_fit_logit2$sims.list$beta_1, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g6 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[1])))

g7 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_2)
             , aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_2, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[2])))

g8 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_2)
             , aes(x=mod_fit_logit2$sims.list$beta_2))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[2])))


bacf <- acf(mod_fit_logit2$sims.list$beta_2, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g9 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[2])))

g10 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_3)
              , aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_3, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[3])))

g11 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_3)
              , aes(x=mod_fit_logit2$sims.list$beta_3))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[3])))


bacf <- acf(mod_fit_logit2$sims.list$beta_3, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g12 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[3])))

g13 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_4)
              , aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_4, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[4])))

g14 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_4)
              , aes(x=mod_fit_logit2$sims.list$beta_4))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[4])))


bacf <- acf(mod_fit_logit2$sims.list$beta_4, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g15 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[4])))

g16 <- ggplot(data=as.data.frame(mod_fit_logit2$sims.list$beta_5)
              , aes(x=seq(1,9000),y=mod_fit_logit2$sims.list$beta_5, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[5])))

g17 <- ggplot(as.data.frame(mod_fit_logit2$sims.list$beta_5)
              , aes(x=mod_fit_logit2$sims.list$beta_5))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[5])))


bacf <- acf(mod_fit_logit2$sims.list$beta_5, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g18 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[5])))

grid.arrange(g1, g3, g2,g4,g6,g5,g7,g9, g8, g10,g12,g11,g13,g15,g14,g16,g18,g17, ncol=3)


union_logit <- as.mcmc(cbind(beta_0=mod_fit_logit2$sims.list$beta_0
                       ,beta_1=mod_fit_logit2$sims.list$beta_1
                       ,beta_2=mod_fit_logit2$sims.list$beta_2
                       ,beta_3=mod_fit_logit2$sims.list$beta_3
                       ,beta_4=mod_fit_logit2$sims.list$beta_4
                       ,beta_5=mod_fit_logit2$sims.list$beta_5))

# Correlation matrix of parameters
data_corr <- as.data.frame(union_logit)
names(data_corr) <- c("beta_0","beta_1","beta_2","beta_3","beta_4","beta_5")
cormat <- round(cor(data_corr),2)
cormat

effectiveSize(union_logit)

geweke_df <- rbind(geweke.diag(union_logit)$z,pnorm(abs(geweke.diag(union_logit)$z),lower.tail=FALSE)*2)
geweke_df

par(mar=c(2,2,2,2))
par(mfrow=c(1,1))
geweke.plot(union_logit)

raftery.diag(union_logit)
raftery_df <-raftery.diag(union_logit)$resmatrix
raftery_df

heidel.diag(union_logit)







####### Defining second probit model by dropping non-significant covariates #######                                  

# Define model
#Identify filepath of model file
modfile_probit <- tempfile()

#Write model to file
writeLines("
model{

  # Likelihood
  for (i in 1:N){
    y[i] ~ dbern(pi[i])
    
    probit(pi[i]) <- beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i] + beta_4*x4[i] + beta_5*x5[i] 
  }
  
  # Define prior for intercept
  beta_0 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta1 (Pregnancies)
  beta_1 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta2 (Glucose)
  beta_2 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta3 (BloodPressure)
  beta_3 ~ dnorm(0, 1.0E-6)

  # Define prior for beta4 (BMI)
  beta_4 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta5 (DiabetesPedigreeFunction)
  beta_5~ dnorm(0, 1.0E-6)
  

}
", con = modfile_probit)


# Preparing data for JAGS
# Set number of individuals
N <- dim(data)[1]

# Response variable
y <- as.vector(as.numeric(as.character(data$Outcome))) 

data_jags2 <- list("y" = y, "N" = N, "x1" = data$Pregnancies
                   , "x2" = data$Glucose
                   , "x3" = data$BloodPressure
                   , "x4" = data$BMI
                   , "x5" = data$DiabetesPedigreeFunction
)

# Defining parameters of interest
parameters2 <- c("beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5")

# Fit model with JAGS
set.seed(seed)
mod_fit_probit <- jags(data = data_jags2,                                               
                       model.file = modfile_probit,                                             
                       parameters.to.save = parameters2,                                
                       n.chains = 1, n.iter = 10000, n.burnin = 1000)

# Get AIC of the model
mod_fit_probit$DIC

# Get summary of the model
mod_fit_probit$summary

g1 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_0), aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_0, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[0])))


g2 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_0), aes(x=mod_fit_probit$sims.list$beta_0))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[0])))


bacf <- acf(mod_fit_probit$sims.list$beta_0, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g3 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+labs(title=expression(paste("Autocorrelogram of ",beta[0])))

g4 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_1)
             , aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_1, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[1])))

g5 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_1)
             , aes(x=mod_fit_probit$sims.list$beta_1))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[1])))

bacf <- acf(mod_fit_probit$sims.list$beta_1, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g6 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[1])))

g7 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_2)
             , aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_2, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[2])))

g8 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_2)
             , aes(x=mod_fit_probit$sims.list$beta_2))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[2])))


bacf <- acf(mod_fit_probit$sims.list$beta_2, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g9 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[2])))

g10 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_3)
              , aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_3, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[3])))

g11 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_3)
              , aes(x=mod_fit_probit$sims.list$beta_3))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[3])))


bacf <- acf(mod_fit_probit$sims.list$beta_3, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g12 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[3])))

g13 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_4)
              , aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_4, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[4])))

g14 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_4)
              , aes(x=mod_fit_probit$sims.list$beta_4))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[4])))


bacf <- acf(mod_fit_probit$sims.list$beta_4, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g15 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[4])))

g16 <- ggplot(data=as.data.frame(mod_fit_probit$sims.list$beta_5)
              , aes(x=seq(1,9000),y=mod_fit_probit$sims.list$beta_5, group=1)) +
  geom_line()+ xlab("Iterations")+ ylab("value")+
  labs(title=expression(paste("",beta[5])))

g17 <- ggplot(as.data.frame(mod_fit_probit$sims.list$beta_5)
              , aes(x=mod_fit_probit$sims.list$beta_5))+
  geom_density()+xlab("value")+ ylab("")+
  labs(title=expression(paste("Density of ",beta[5])))


bacf <- acf(mod_fit_probit$sims.list$beta_5, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))

g18 <- ggplot(data=bacfdf, mapping=aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity")+
  xlab("Lag")+ ylab("ACF")+
  labs(title=expression(paste("Autocorrelogram of ",beta[5])))

grid.arrange(g1, g3, g2,g4,g6,g5,g7,g9, g8, g10,g12,g11,g13,g15,g14,g16,g18,g17, ncol=3)


union_probit <- as.mcmc(cbind(beta_0=mod_fit_probit$sims.list$beta_0
                             ,beta_1=mod_fit_probit$sims.list$beta_1
                             ,beta_2=mod_fit_probit$sims.list$beta_2
                             ,beta_3=mod_fit_probit$sims.list$beta_3
                             ,beta_4=mod_fit_probit$sims.list$beta_4
                             ,beta_5=mod_fit_probit$sims.list$beta_5))

# Correlation matrix of parameters
data_corr <- as.data.frame(union_probit)
names(data_corr) <- c("beta_0","beta_1","beta_2","beta_3","beta_4","beta_5")
cormat <- round(cor(data_corr),2)
cormat

effectiveSize(union_probit)

geweke_df <- rbind(geweke.diag(union_probit)$z,pnorm(abs(geweke.diag(union_probit)$z),lower.tail=FALSE)*2)
geweke_df

par(mar=c(2,2,2,2))
par(mfrow=c(1,1))
geweke.plot(union_probit)

raftery.diag(union_probit)
raftery_df <-raftery.diag(union_probit)$resmatrix
raftery_df

heidel.diag(union_probit)

# Fitting Frequentist logistic regression model
fre_mod_fit_logit <- glm(factor(Outcome)~Pregnancies+Glucose+BloodPressure+BMI+DiabetesPedigreeFunction,data = data,family = binomial())
summary(fre_mod_fit_logit)
AIC(fre_mod_fit_logit)



## Checking ability of the model to recover true parameters
# Define true parameters
b0 <- -4.60
b1 <- 0.09
b2 <- 0.01
b3 <- -0.006
b4 <- 0.05
b5 <- 0.47

n_sim <- 10000
set.seed(seed)
# x1 <- rbeta(n = n_sim,shape1 = 2,shape2 = 0.5)*100
x1 <- data$Pregnancies
set.seed(seed)
# x2 <- rnormmix(n_sim,lambda = c(0.3, 0.4, 0.3), mu = c(2.8, 5.6, 8.4), sigma = c(0.65, 0.92,0.55) )
x2 <- data$Glucose
set.seed(seed)
# x3 <- rgamma(n = n_sim,shape = 1.2,rate = 1/2000)+4000
x3 <- data$BloodPressure
set.seed(seed)
# x4 <- rgamma(n = n_sim,shape = 5,rate = 1/15)+40
x4 <- data$BMI
set.seed(seed)
# x5 <- rgamma(n = n_sim,shape = 0.85,rate = 1/5500)
x5 <- data$DiabetesPedigreeFunction

z_sim <- b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5

pi_sim <- 1/(1+exp(-z_sim))

set.seed(seed)
y_sim <- sapply(pi_sim,function(z){rbinom(1,1,z)})

ggplot(as.data.frame(y_sim), aes(x=y_sim))+
  geom_density()+xlab("Y")+ ylab("")+
  labs(title=expression(paste("Distribution for Y simulated")))+
  scale_x_continuous(limits = c(-0.5, 1.5))


#Identify filepath of model file
sim_modfile_logit2 <- tempfile()

#Write model to file
writeLines("
model{

  # Likelihood
  for (i in 1:N){
    y[i] ~ dbern(pi[i])
    
    logit(pi[i]) <- beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i] + beta_4*x4[i] + beta_5*x5[i] 
  }
  
  # Define prior for intercept
  beta_0 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta1 (Pregnancies)
  beta_1 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta2 (Glucose)
  beta_2 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta3 (BloodPressure)
  beta_3 ~ dnorm(0, 1.0E-6)

  # Define prior for beta4 (BMI)
  beta_4 ~ dnorm(0, 1.0E-6)
  
  # Define prior for beta5 (DiabetesPedigreeFunction)
  beta_5~ dnorm(0, 1.0E-6)
  

}
", con = sim_modfile_logit2)


# Preparing data for JAGS
# Set number of individuals
N <- dim(data)[1]

data_jags2_sim <- list("y" = y_sim, "N" = N, "x1" = x1
                   , "x2" = x2
                   , "x3" = x3
                   , "x4" = x4
                   , "x5" = x5
)

# Defining parameters of interest
parameters2 <- c("beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5")

# Fit model with JAGS
set.seed(seed)
mod_fit_logit2_sim <- jags(data = data_jags2_sim,                                               
                       model.file = sim_modfile_logit2,                                             
                       parameters.to.save = parameters2,                                
                       n.chains = 1, n.iter = 10000, n.burnin = 1000)

# Get AIC of the model
mod_fit_logit2_sim$DIC

# Get summary of the model
mod_fit_logit2_sim$summary