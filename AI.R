#--------Title:  Predicting energy commodity prices amidst -------------
#-------worldwide energy transitions using deep learning models --
##library#####
library(dplyr)
library(tidyr)
library(keras)
library(neuralnet)
library(zoo)
library(caret)
library(NeuralNetTools)
library(tidyr)
library(tensorflow)
library(readxl)
library(xts)
library(writexl)
library(xgboost)
library(stats)
library(ggplot2)
library(fdesc)
library(mltsp)
library(e1071)
library(flexsurv)
library(reticulate)
library(forecast)
#************import data********************-----
excel_file <- "C:/Users/Foued Azuz 14/OneDrive/AI energy/dataset.xlsx"
sheet_names <- excel_sheets(excel_file)
all_dataframes <- lapply(sheet_names, function(sheet) {
    read_excel(excel_file, sheet = sheet)
  })

all_dataframes <- lapply(all_dataframes, as.data.frame)
  
for (i in 1:22) {
    
    all_dataframes[[i]][["date"]] <- as.Date(all_dataframes[[i]][["date"]])

}
merged_df <- all_dataframes[[1]]
  for (i in 2:length(all_dataframes)) {
    df <- all_dataframes[[i]]
      merged_df <- merge(merged_df, df, by = 'date', all = TRUE)
  }
  
#merged_df$date <- as.Date(merged_df$date)
#merged_df <- merged_df[!is.na(merged_df$date), ]
xts_obj <- xts(merged_df[, -1], order.by = merged_df$date)
write_xlsx(merged_df, "file.xlsx")
#---------------------plot data--------------------------

ecp <-na.omit(merged_df[,c(1,2,3,4)])  #energy commodities prices
colnames(ecp)<- c("Date","OIL","GAS","Coal")


df <- ecp %>%
  gather(key = "Variable", value = "value", -Date)
ggplot(df, aes(x = Date, y = value, linetype = Variable)) + 
  geom_line(aes(color = Variable), size = 1) +
  scale_color_manual(values = c("#59504E", "#59504E", "#222222")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Commodity", linetype = "Commodity")+
theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))




#------------prepare the dataset for missing value--------------####
column_names <- colnames(merged_df)
start_date <- as.Date("2006-10-16")
end_date <- as.Date("2023-07-24")
mask <- merged_df$date >= start_date & merged_df$date <= end_date
subset_df <- merged_df[mask, ]
subset_df_xts<-as.xts(subset_df)

##Replace missing value#####


fill_and_merge <- function(df, variable_name) {
  zoo_variable <- zoo(df[, variable_name], order.by = df$date)
  filled_zoo_variable <- na.locf(zoo_variable, fromLast = TRUE)
  
  filled_df <- data.frame(Date = index(filled_zoo_variable), 
                          value = coredata(filled_zoo_variable))
  colnames(filled_df) <- c("date", variable_name)
  
  df <- merge(df, filled_df, by = "date", all.x = TRUE)
  col_to_remove <- paste(variable_name, ".x", sep = "")
  df[[col_to_remove]] <- NULL
  col_to_rename <- variable_name
  df[[col_to_rename]] <- df[[paste(variable_name, ".y", sep = "")]]
  df[[paste(variable_name, ".y", sep = "")]] <- NULL
  
  return(df)
}

variables_list <- c("solarthermal", "solarpvthermalhybrid", "solarpv","wind",
                    "hydropower", "marineandtidal","bioenergy","geothermal",
                    "bioenergyLCOE","geothermalLCOE",	"offshorewindLCOE",
                    "solarphotovoltaicLCOE","concentratedsolarLCOE",
                    "hydropowerLCOE","onshorewindLCOE", "AC_HYD_E",
                    "AC_SOLAR_E","AC_WIND_E","GR","EPI","TEMP",
                    "EEPH","co2_per_capita",
                    "TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP")


for (variable in variables_list) {
  subset_df <- fill_and_merge(subset_df, variable)
}

#end



#export summary statistics####

subset_df_xts <- as.xts(subset_df)
mytable <- desc(subset_df_xts)
export(mytable)


#define functions ------------------------------
RMSE <- function (true_values,predictions) {
  squared_diff <- (true_values - predictions)^2
  
  mean_squared_diff <- mean(squared_diff)
  
  rmse <- sqrt(mean_squared_diff)
  
  cat("Root Mean Squared Error (RMSE):", rmse, "\n")
  
}



Rsquared <- function(true_values,predictions) {
  correlation <- cor(true_values, predictions)
  rsquared <- correlation^2
  
  cat("R-squared (R²):", rsquared, "\n")
  
}



MAE <- function (true_values,predictions) {
  
  abs_diff <- abs((true_values - predictions)/true_values)
  mad <- mean(abs_diff)
  cat("Mean Absolute Deviation (MAD):", mad, "\n") 
  
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

inverse_normalize <- function(y, original_data) {
  min_original <- min(original_data)
  max_original <- max(original_data)
  return (y * (max_original - min_original) + min_original)
}



#******************NARX model******************####

# ***NARX model using neuralnet for WTI####

exoge <-subset_df_xts[,-1]
wti<-subset_df_xts$wti
data <- data.frame(wti, exoge)
data <- data[complete.cases(data$wti), ] #delete missing value dor wti only
wti <- na.omit(wti)
lagged <- embed(wti, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("wti","wti1","wti2","wti3","wti4")
time <- length(data$wti)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))

maxmindata <- maxmindata[,-c(2,3)]

## Split the data into training and testing sets
n<-length(maxmindata$wti)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#
## for all factors
formula <- wti ~ wti1+wti2+wti3+wti4+TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors


# ## for RE EVNT factors only
# formula <- wti ~ wti1 + wti2+wti3+wti4+TEMP+co2_per_capita+#envt factors
#   solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
#   bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
#   hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
#   GR+EPI# political factors


### Choose the best model##


grid <-  expand.grid(layer1 = c(2,3,4),
                     layer2 = c(2,3,4),
                     layer3 = c(2,3,4))

###with all variables##
library(caret)
nnwti <- train(formula, 
            data = training_data, 
            method = "neuralnet", 
            tuneGrid = grid,
            metric = "RMSE",
            preProc = c("center", "scale", "nzv","pca"), #good idea to do this with neural nets - your error is due to non scaled data
            trControl = trainControl(
              method = "cv",
              number = 5,
              verboseIter = TRUE)
)

nnwti





# x.interest <- testing_data[,-1]
# shapley <- Shapley$new(mod, x.interest = x.interest)
# plot(shapley)
# 


#find the mean impact value

# n<-length(maxmindata$wti)
# train_frac <- 0.8
# train_idx <- 1:round(train_frac * n)
# X_train <- maxmindata[train_idx, -c(1,2,3)]
# y_train <- maxmindata[train_idx, 1]
# X_test <- maxmindata[-train_idx, -c(1,2,3)]
# y_test <- maxmindata[-train_idx, 1]
# 
# miv_values <- numeric(ncol(X_train))
# for (i in 1:ncol(X_train)) {
#   nn_model_without_i <- neuralnet(wti ~ . - X_train[, i], data = X_train, hidden = c(5, 3))
#   nn_predictions_without_i <- predict(nn_model_without_i, newdata = X_test)
#   miv_values[i] <- mean((y_test - nn_predictions_without_i)^2)
# }
# 



#plot neural diagram
tiff("nnfig.jpg",width = 10, height = 10, units = 'in', res = 350) #for high resolution
plotnet(nnn$finalModel,cex_val =0.5,"","WTI",bord_col="black",pad_x=0.75,
       bias=FALSE)
title("Oil wti")
dev.off()



#plot predictions vs actual
predictions <-  predict(nnwti1, newdata = testing_data)

tiff("pred_act narx.jpg",width = 10, height = 5, units = 'in', res = 350) #for high resolution
plot(testing_data$wti, predictions, 
     main = "Actual vs. Predicted",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1, col = "red") 
dev.off()
# plot predictions and actual

data_xts <- as.xts(datad)
maxmindata_xts <-lapply(data_xts, normalize)
wti_xts <- xts(maxmindata_xts$wti, order.by = index(maxmindata_xts[["wti"]]))
wti_xts$predictions <- 0
wti_xts$predictions[test_idx,] <- predictions
wtidf <- as.data.frame(wti_xts)
wtidf$Date <- index(wti_xts)
colnames(wtidf) <- c("Actual","prediction","Date")

df <- wtidf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("predi_narx.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))

dev.off()

##Gevrey####
library(caret)
importance_scores <- varImp(nnwti)
variable_importance <- importance_scores$importance
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI"),
  pastvalue.F =c("wti1","wti2","wti3","wti4")
  
  
)
grouped_importance <- lapply(groups, function(group)
  sum(variable_importance[rownames(variable_importance) %in% group, "Overall"]))

library(ggplot2)

group_names <- names(grouped_importance)
importance_values <- unlist(grouped_importance)
df <- data.frame(Group = group_names, Importance = importance_values)


tiff("importancewtiGEVREY.jpg",width = 6, height = 3, units = 'in', res = 350)
g1grevrey <- ggplot(df, aes(x = Importance, y = Group)) +
  geom_bar(stat = "identity", fill = "grey50") +
  xlab("Importance") +
  ylab("Group") +
  ggtitle("Grouped Importance Values for WTI") +
  theme_minimal()
dev.off()



## fisher####

library(iml)
testing_data <- testing_data[,-c(2,3)]
mod <- Predictor$new(nnwti, data = testing_data[,-1], y = testing_data$wti)

# We can calculate joint importance of groups of features
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI")
  
  
)
imp <- FeatureImp$new(mod, loss = "mae", features = groups)
imp2 <- FeatureImp$new(mod, loss = "mae")

fv <- as.data.frame(imp2$results)
colnames(fv) <- c("Feature","i95","Importance","i95","error")
fv <- fv[,c(1,3)]
colnames(fv) <- c("Feature","Importance")


x <- c("AC_HYD","P.BIO","P.SOL-THER","AC_SOLAR","P.WIND","P.SHPT","LTY","GCOP",
                    "P.SOLAR","Hydropower","GR","P.MT","Concentrated solar","L1.WTI","L3.WTI"
       ,"L4.WTI","Geothermal","L2.WTI","Energy demand","Wind energy","CO2",
                    "TB","SRV","GIPI","Solar photovoltaic","EPI",
               "Bioenergy","UMP","M2","IPI","Offshore wind","P.HYD",
                 "AC_WIND","P.GEO","TEMP","I")

fv$Feature <- factor(x)

library(patchwork)
tiff("importancewti.jpg",width = 12, height = 4, units = 'in', res = 350)
plot(imp) + gridExtra::tableGrob(fv[1:10, c('Feature', 'Importance')])
dev.off()



# ***NARX model using neuralnet for gas####

exoge <-subset_df_xts[,-2]
gp<-subset_df_xts$GP
data <- data.frame(gp, exoge)
data <- data[complete.cases(data$GP), ] #delete missing value dor wti only
gp <- na.omit(gp)
lagged <- embed(gp, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("gp","gp1","gp2","gp3","gp4")
time <- length(data$GP)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))




## Split the data into training and testing sets
n<-length(maxmindata$GP)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#

## for all factors
formula <- GP ~ gp1+gp2+gp3+gp4+TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors



### Choose the best model##


grid <-  expand.grid(layer1 = c(2,3,4),
                     layer2 = c(2,3,4),
                     layer3 = c(2,3,4))

###with all variables##
nnGP <- train(formula, 
              data = training_data, 
              method = "neuralnet", 
              tuneGrid = grid,
              metric = "RMSE",
              preProc = c("center", "scale", "nzv","pca"), #good idea to do this with neural nets - your error is due to non scaled data
              trControl = trainControl(
                method = "cv",
                number = 5,
                verboseIter = TRUE)
)

nnGP  # find the best model


#plot neural diagram
tiff("nnfigGP.jpg",width = 10, height = 10, units = 'in', res = 350) #for high resolution
plotnet(nnGP$finalModel,cex_val =0.5,"","GP",bord_col="black",pad_x=0.75,
        bias=FALSE)
title("Natural Gas")
dev.off()


# plot predictions vs actual
predictions <-  predict(nnGP, newdata = testing_data)
tiff("pred_narx_GP.jpg",width = 10, height = 5, units = 'in', res = 350) #for high resolution
plot(testing_data$GP, predictions, 
     main = "Actual vs. Predicted",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1, col = "red") 
dev.off()

# plot predictions and actual
library(xts)
data_xts <- as.xts(datad)
maxmindata_xts <-lapply(data_xts, normalize)
GP_xts <- xts(maxmindata_xts$GP, order.by = index(maxmindata_xts[["GP"]]))
GP_xts$predictions <- 0
GP_xts$predictions[test_idx,] <- predictions
GPdf <- as.data.frame(GP_xts)
GPdf$Date <- index(GP_xts)
colnames(GPdf) <- c("Actual","prediction","Date")
merged <- GPdf[test_idx,]

df <- GPdf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("NARX_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))

dev.off()


##Gevrey####
library(caret)
importance_scores <- varImp(nnGP)
variable_importance <- importance_scores$importance
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI"),
  pastvalue.F =c("gp1","gp2","gp3","gp4")
  
  
)
grouped_importance <- lapply(groups, function(group)
  sum(variable_importance[rownames(variable_importance) %in% group, "Overall"]))

library(ggplot2)

group_names <- names(grouped_importance)
importance_values <- unlist(grouped_importance)
df <- data.frame(Group = group_names, Importance = importance_values)


tiff("importancegpGEVREY.jpg",width = 6, height = 3, units = 'in', res = 350)
g2grevrey <- ggplot(df, aes(x = Importance, y = Group)) +
  geom_bar(stat = "identity", fill = "grey50") +
  xlab("Importance") +
  ylab("Group") +
  ggtitle("Grouped Importance Values for Natural Gas") +
  theme_minimal()
dev.off()

## ifisher####

library(iml)
testing_data <- testing_data[,-c(2,3)]
mod <- Predictor$new(nnGP, data = testing_data[,-1], y = testing_data$GP)

# We can calculate joint importance of groups of features
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI")
  
  
)
imp <- FeatureImp$new(mod, loss = "mae", features = groups)
imp2 <- FeatureImp$new(mod, loss = "mae")

fv <- as.data.frame(imp2$results)
colnames(fv) <- c("Feature","i95","Importance","i95","error")
fv <- fv[,c(1,3)]
colnames(fv) <- c("Feature","Importance")


x <- c("CO2","L1.GP","L2.GP","I","L3.GP","L4.GP","P.BIO","P.SOL-THER","GCOP","P.MT",
       "P.SHPT","Hydropower","P.HYD","AC_SOLAR","SRV","Offshore wind","AC_HYD","P.GEO",
       "AC_WIND","TB","Energy demand","P.SOLAR","TEMP","IPI","Geothermal","M2","EPI",
       "GR","Bioenergy","P.WIND","UMP","Wind energy", "GIPI","Solar photovoltaic",
       "LTY","Concentrated solar")

fv$Feature <- factor(x)

library(patchwork)
tiff("importancegp.jpg",width = 12, height = 4, units = 'in', res = 350)
plot(imp) + gridExtra::tableGrob(fv[1:10, c('Feature', 'Importance')])
dev.off()







# ***NARX model using neuralnet for Coal####

exoge <-subset_df_xts[,-3]
coal<-subset_df_xts$coal
data <- data.frame(coal, exoge)
data <- data[complete.cases(data$coal), ] #delete missing value dor wti only
coal <- na.omit(coal)
lagged <- embed(coal, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("coal","coal1","coal2","coal3","coal4")
time <- length(data$coal)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))


## Split the data into training and testing sets
n<-length(maxmindata$coal)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#
## for all factors
formula <- coal ~ coal1+coal2+coal3+coal4+TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors

###with all variables##
nncoal <- train(formula, 
             data = training_data, 
             method = "neuralnet", 
             tuneGrid = grid,
             metric = "RMSE",
             preProc = c("center", "scale", "nzv","pca"), #good idea to do this with neural nets - your error is due to non scaled data
             trControl = trainControl(
               method = "cv",
               number = 5,
               verboseIter = TRUE)
)

nncoal



#plot neural diagram
tiff("nncoal.jpg",width = 10, height = 10, units = 'in', res = 350) #for high resolution
plotnet(nncoal$finalModel,cex_val =0.5,"","Coal",bord_col="black",pad_x=0.75,
        bias=FALSE)
title("Coal")
dev.off()



#plot predictions vs actual
predictions <-  predict(nncoal, newdata = testing_data)

tiff("pred_narx_coal.jpg",width = 10, height = 5, units = 'in', res = 350) #for high resolution
plot(testing_data$coal, predictions, 
     main = "Actual vs. Predicted",
     xlab = "Actual",
     ylab = "Predicted")
abline(0, 1, col = "red") 
dev.off()
# plot predictions and actual
library(xts)
data_xts <- as.xts(datad)
maxmindata_xts <-lapply(data_xts, normalize)
coal_xts <- xts(maxmindata_xts$coal, order.by = index(maxmindata_xts[["coal"]]))
coal_xts$predictions <- 0
coal_xts$predictions[test_idx,] <- predictions
coaldf <- as.data.frame(coal_xts)
coaldf$Date <- index(coal_xts)
colnames(coaldf) <- c("Actual","prediction","Date")
library(tidyr)
library(dplyr)
df <- coaldf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("predi_narx_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))

dev.off()

## Gevrey####
library(caret)
importance_scores <- varImp(nncoal)
variable_importance <- importance_scores$importance
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI"),
  pastvalue.F =c("coal1","coal2","coal3","coal4")
  
  
)
grouped_importance <- lapply(groups, function(group)
  sum(variable_importance[rownames(variable_importance) %in% group, "Overall"]))

library(ggplot2)

group_names <- names(grouped_importance)
importance_values <- unlist(grouped_importance)
df <- data.frame(Group = group_names, Importance = importance_values)


tiff("importancecoalGEVREY.jpg",width = 6, height = 3, units = 'in', res = 350)
g3grevrey <- ggplot(df, aes(x = Importance, y = Group)) +
  geom_bar(stat = "identity", fill = "grey50") +
  xlab("Importance") +
  ylab("Group") +
  ggtitle("Grouped Importance Values for Coal") +
  theme_minimal()
dev.off()



## fisher####

library(iml)
testing_data <- testing_data[,-c(2,3)]
mod <- Predictor$new(nncoal, data = testing_data[,-1], y = testing_data$coal)

# We can calculate joint importance of groups of features
groups = list(
  Macroeconomic.F = c("TB","LTY","I","SRV","GOP","GIPIO","M2","IPI","UMP","EEPH"),
  Environement.F = c("TEMP","co2_per_capita"),
  Technological.F = c("solarpv","solarthermal","solarpvthermalhybrid","wind",
                      "hydropower","marineandtidal","bioenergy","geothermal"),
  RenewableEnergy.F = c("bioenergyLCOE","geothermalLCOE","offshorewindLCOE",
                        "solarphotovoltaicLCOE","concentratedsolarLCOE","hydropowerLCOE",
                        "onshorewindLCOE","AC_WIND_E","AC_SOLAR_E","AC_HYD_E"),
  GeopoliticalRisk.F = c("GR","EPI")
  
  
)
imp <- FeatureImp$new(mod, loss = "mae", features = groups)
imp2 <- FeatureImp$new(mod, loss = "mae")

fv <- as.data.frame(imp2$results)
colnames(fv) <- c("Feature","i95","Importance","i95","error")
fv <- fv[,c(1,3)]
colnames(fv) <- c("Feature","Importance")


x <- c("L1.coal","L3.coal","L2.coal","L4.coal","P.SOL-THER","P.GEO","CO2","P.HYD",
       "P.WIND","UMP","P.SOLAR","AC_HYD","TB","P.SHPT","SRV","M2","IPI",
       "Hydropower","Bioenergy", "EPI","GIPI","GCOP","Solar photovoltaic","P.BIO", "Concentrated solar",
         "Geothermal","P.MT","AC_SOLAR","TEMP","Offshore wind","Energy demand",
      "Wind energy", "AC_WIND","LTY","I","GR"          )
  
fv$Feature <- factor(x)

library(patchwork)
tiff("importancecoal.jpg",width = 12, height = 4, units = 'in', res = 350)
plot(imp) + gridExtra::tableGrob(fv[1:10, c('Feature', 'Importance')])
dev.off()






#**********************ANN*****************************####
#*
# ANN model####
#------------------------------------------******************

## ANN using macro factors####
### for WTI
exoge <-subset_df_xts[,-1]
wti<-subset_df_xts$wti
data <- data.frame(wti, exoge)
data <- data[complete.cases(data$wti), ] #delete missing value dor wti only
wti <- na.omit(wti)
lagged <- embed(wti, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("wti","wti1","wti2","wti3","wti4")
time <- length(data$wti)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))


n<-length(maxmindata$wti)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]

formula <- wti ~ TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH#macro factors
 
grid <-  expand.grid(layer1 = c(2,3,4),
                     layer2 = c(2,3,4),
                     layer3 = c(2,3,4))

nnwtimacro <- train(formula, 
               data = training_data, 
               method = "neuralnet", 
               tuneGrid = grid,
               metric = "RMSE",
               preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
               trControl = trainControl(
                 method = "cv",
                 number = 5,
                 verboseIter = TRUE)
)
nnwtimacro


predictions <-  predict(nnwtimacro, newdata = testing_data)

###for natural Gas

exoge <-subset_df_xts[,-2]
gp<-subset_df_xts$GP
data <- data.frame(gp, exoge)
data <- data[complete.cases(data$GP), ] #delete missing value dor wti only
gp <- na.omit(gp)
lagged <- embed(gp, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("gp","gp1","gp2","gp3","gp4")
time <- length(data$GP)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))


n<-length(maxmindata$GP)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]

formula <- GP ~ TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH#macro factors

grid <-  expand.grid(layer1 = c(2,3,4),
                     layer2 = c(2,3,4),
                     layer3 = c(2,3,4))

nngpmacro <- train(formula, 
                    data = training_data, 
                    method = "neuralnet", 
                    tuneGrid = grid,
                    metric = "RMSE",
                    preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
                    trControl = trainControl(
                      method = "cv",
                      number = 5,
                      verboseIter = TRUE)
)
nngpmacro

predictions <-  predict(nngpmacro, newdata = testing_data)

###for coal

exoge <-subset_df_xts[,-3]
coal<-subset_df_xts$coal
data <- data.frame(coal, exoge)
data <- data[complete.cases(data$coal), ] #delete missing value dor wti only
coal <- na.omit(coal)
lagged <- embed(coal, 5)
data <- data[-c(1:8),]
lagged <- lagged[-c(1:4),]
colnames(lagged) <- c("coal","coal1","coal2","coal3","coal4")
time <- length(data$coal)
lagged <- lagged[,-1]
datad <- cbind(data,lagged)
datad[] <- lapply(datad, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(datad, normalize))


n<-length(maxmindata$coal)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]

formula <- coal ~ TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH#macro factors

grid <-  expand.grid(layer1 = c(2,3,4),
                     layer2 = c(2,3,4),
                     layer3 = c(2,3,4))

nncoalmacro <- train(formula, 
                   data = training_data, 
                   method = "neuralnet", 
                   tuneGrid = grid,
                   metric = "RMSE",
                   preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
                   trControl = trainControl(
                     method = "cv",
                     number = 5,
                     verboseIter = TRUE)
)
nncoalmacro

predictions <-  predict(nncoalmacro, newdata = testing_data)



# plot predictions and actual
library(xts)
data_xts <- as.xts(datad)
maxmindata_xts <-lapply(data_xts, normalize)
coal_xts <- xts(maxmindata_xts$coal, order.by = index(maxmindata_xts[["coal"]]))
coal_xts$predictions <- 0
coal_xts$predictions[test_idx,] <- predictions
coaldf <- as.data.frame(coal_xts)
coaldf$Date <- index(coal_xts)
colnames(coaldf) <- c("Actual","prediction","Date")
#DW test
bb <- coaldf[test_idx,]
bb$error <- (bb$Actual-bb$prediction)^2








##ANN using past value####

wti <- subset_df_xts[,1]
wti <- na.omit(wti)
lagged_data <- embed(wti, 5)
input_data <- lagged_data[, -1 ]
output_data <- lagged_data[, 1]
ann_data <- data.frame(input_data, output_data)
ann_datascale <- as.data.frame(lapply(ann_data, normalize))
#train the model
n<-length(ann_datascale$output_data)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- ann_datascale[train_idx,]
testing_data <- ann_datascale[test_idx,]

grid <-  expand.grid(layer1 = c(1,2,3),
                     layer2 = c(1,2,3),
                     layer3 = c(1,2,3))


formula <- output_data ~ X1+X2+X3+X4
  

ann <- train(formula, 
                data = training_data, 
                method = "neuralnet", 
                tuneGrid = grid,
                metric = "RMSE",
                preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
                trControl = trainControl(
                  method = "cv",
                  number = 5,
                  verboseIter = TRUE)
)
ann
#prepare for plot
predictions <- predict(nngpmacro, testing_data)

RMSE(testing_data$output_data,predictions)
MAE(testing_data$output_data,predictions)
Rsquared(testing_data$output_data,predictions)


#|||||||||||||||||||||||||||||||||||
actual <- as.data.frame(ann_data$output_data) # find equivalent 
actual <- actual[test_idx,]
revpredict <- inverse_normalize(predictions,actual)
mergedwti <- as.data.frame(cbind(actual,revpredict))

wtiB <- subset_df_xts$wti
wtiB <- na.omit(wti)
split_index <- floor(0.8 * nrow(wtiB))
wtiB$predict <- 0
wtiB$predict[(split_index)+1:nrow(wtiB)] <- revpredict
wtidf <- as.data.frame(wtiB)
wtidf$date <- index(wtiB)
colnames(wtidf) <- c("Actual","prediction","Date")

df <- wtidf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("ANN_pred_wti.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g1 <-ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

dev.off()


# ANN model for gas###
#------------------------------------------******************

gp <- subset_df_xts[,2]
gp <-na.omit(gp)
lagged_data <- embed(gp, 5)
input_data <- lagged_data[, -1 ]
output_data <- lagged_data[, 1]
ann_data <- data.frame(input_data, output_data)
ann_datascale <- as.data.frame(lapply(ann_data, normalize))

#train the model
n<-length(ann_datascale$output_data)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- ann_datascale[train_idx,]
testing_data <- ann_datascale[test_idx,]

grid <-  expand.grid(layer1 = c(1, 2,3),
                     layer2 = c(1, 2,3),
                     layer3 = c(1,2,3))


formula <- output_data ~ X1+X2+X3+X4

#ann for gas
ann1 <- train(formula, 
              data = training_data, 
              method = "neuralnet", 
              tuneGrid = grid,
              metric = "RMSE",
              preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
              trControl = trainControl(
                method = "cv",
                number = 5,
                verboseIter = TRUE)
)
ann1

#prepare for plot
predict_gp_ann <- predict(ann1, testing_data)

RMSE(testing_data$output_data,predict_gp_ann)
MAE(testing_data$output_data,predict_gp_ann)
Rsquared(testing_data$output_data,predict_gp_ann)



actual <- as.data.frame(ann_data$output_data) # find equivalent 
actual <- actual[test_idx,]
revpredict <- inverse_normalize(predict_gp_ann,actual)
mergedgp <- as.data.frame(cbind(actual,revpredict))

gp <- subset_df_xts$GP
gp <- na.omit(gp)
split_index <- floor(0.8 * nrow(gp)+1)
gp$predict <- 0
gp$predict[(split_index + 1):nrow(gp)] <- revpredict
gpdf <- as.data.frame(gp)
gpdf$date <- index(gp)
colnames(gpdf) <- c("Actual","prediction","Date")

df <- gpdf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("ANN_pred_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g2 <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

dev.off()



# ANN model using neuralnet for coal##
coal <- subset_df_xts[,3]
coal <- na.omit(coal)
lagged_data <- embed(coal, 5)
input_data <- lagged_data[, -1 ]
output_data <- lagged_data[, 1]
ann_data <- data.frame(input_data, output_data)
ann_datascale <- as.data.frame(lapply(ann_data, normalize))

#train the model
n<-length(ann_datascale$output_data)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- ann_datascale[train_idx,]
testing_data <- ann_datascale[test_idx,]

grid <-  expand.grid(layer1 = c(1, 2,3),
                     layer2 = c(1, 2,3),
                     layer3 = c(1,2,3))


formula <- output_data ~ X1+X2+X3+X4

#ann for coal
ann2 <- train(formula, 
              data = training_data, 
              method = "neuralnet", 
              tuneGrid = grid,
              metric = "RMSE",
              preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
              trControl = trainControl(
                method = "cv",
                number = 5,
                verboseIter = TRUE)
)
ann2

#prepare for plot

predict_coal_ann <- predict(ann2, testing_data)

RMSE(testing_data$output_data,predict_coal_ann)
MAE(testing_data$output_data,predict_coal_ann)
Rsquared(testing_data$output_data,predict_coal_ann)



actual <- as.data.frame(ann_data$output_data) # find equivalent 
actual <- actual[test_idx,]
revpredict <- inverse_normalize(predict_coal_ann,actual)
mergedcoal <- as.data.frame(cbind(actual,revpredict))



coalB <- subset_df_xts$coal
coalB <- na.omit(coalB)
split_index <- floor(0.8 * nrow(coalB)+1)
coalB$predict <- 0
coalB$predict[(split_index + 1):nrow(coalB)] <- revpredict
coaldf <- as.data.frame(coalB)
coaldf$date <- index(coalB)
colnames(coaldf) <- c("Actual","prediction","Date")

df <- coaldf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("ANN_pred_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g3 <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

dev.off()


graph <- (g1+g2)/g3 +plot_layout(guides="collect")

# Save the combined plot as a PNG file
ggsave("combined_plot.png", width = 12, height = 6, plot = graph, dpi = 300)


#**********************LSTM********************************####
#LSTM for WTI, Gp and coal  change each name accordingly ####
wti <- subset_df_xts[,1]
wti <- na.omit(wti)
wti_diff =as.numeric(diff(wti, differences = 1))
gp <- subset_df_xts[,2]
gp <- na.omit(gp)
gp_diff <- as.numeric(diff(gp, differences = 1))


coal <- subset_df_xts[,3]
coal <- na.omit(coal)

coal_diff <- as.numeric(diff(coal, differences = 1))
coal_diff <- na.omit(coal_diff)
##----------------------------"fill the code"--------------------------------------------

lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised = lag_transform(coal_diff, 1)
N = nrow(supervised)
n = round(N *0.8, digits = 0)
train = supervised[1:n, ]
test  = supervised[(n+1):N,  ]
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}

Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled[["scaled_train"]][["x"]]
x_train = Scaled[["scaled_train"]][["x-1"]]

y_test =Scaled[["scaled_test"]][["x"]]
x_test =Scaled[["scaled_test"]][["x-1"]]

invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}
dim(x_train) <- c(length(x_train), 1, 1)

X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1                # must be a common factor of both the train and test samples
units = 1     
library(keras)
model <- keras_model_sequential()

model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),  
  metrics = c("mean_absolute_error")
)
summary(model)

Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states() 
}

L = length(x_test)
scaler = Scaled$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + coal[(n+i)]
  # store
  predictions[i] <- yhat
}
#----------------------------"end of code"-----------------------------------


# plot predictions vs actual
wti <- subset_df_xts[,1]
wti <- na.omit(wti)
coal <- subset_df_xts[,3]
coal <- na.omit(coal)
gp <- subset_df_xts[,2]
gp <- na.omit(gp)

act_pred <- coal[(n+1):N,] # find the original value equivalent to predictions
combin <- cbind(act_pred,predictions)
colnames(combin) <-c("actual","predictions")
combin <- as.data.frame(combin)
combin <- na.omit(combin)


rmse <- sqrt(mean((combin$actual - combin$predictions)^2))
mae <- mean(abs((combin$actual - combin$predictions)/combin$actual))
r <- cor(combin$predictions,combin$actual)^2
#repeate before running code
wti <- subset_df_xts[,1]
wti <- na.omit(wti)
act_pred <- wti[(n+1):N,] # find the original value equivalent to predictions
coal <- subset_df_xts[,3]
coal <- na.omit(coal)
gp <- subset_df_xts[,2]
gp <- na.omit(gp)

split_index <- floor(0.8 * nrow(gp))
gp$predict <- 0
gp$predict[(split_index + 1):nrow(gp)] <- predictions
gpdf <- as.data.frame(gp)
library(xts)
gpdf$date <- index(gp)
colnames(gpdf) <- c("Actual","prediction","Date")
gpdf <- na.omit(gpdf)

library(tidyr)
library(dplyr)

df <- gpdf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

# tiff("LSTM_pred_gp.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution
library(ggplot2)
g1lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))




g2lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))


g3lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

# dev.off()



library(patchwork)
graphLSTM <- (g1lstm+g2lstm)/g3lstm +plot_layout(guides="collect")

# Save the combined plot as a PNG file
ggsave("combined_plot_LSTM.png", width = 12, height = 6, plot = graphLSTM, dpi = 300)














#*****************xgboost**********************####
##for WTI####
#------------------------------------------******************
wtiB <- subset_df_xts$wti
wtiB <-na.omit(wtiB)
wtidf <- as.data.frame(wtiB)
wtidf$date <- index(wtiB)
wtiBoost <- wtidf %>%
  mutate(Lag1 = lag(wti),
         Lag2 = lag(wti, 2))
wtiBoost <- na.omit(wtiBoost)

train_size <- 0.8
train_index <- 1:(train_size * nrow(wtiBoost))

train_data <- wtiBoost[train_index, ]
test_data <- wtiBoost[-train_index, ]

# Train the XGBoost Model
params <- list(
  objective = "reg:squarederror",  # Regression task
  eval_metric = "rmse"  # Root Mean Squared Error as the evaluation metric
)

label_column <- as.numeric(train_data$wti)
features <- as.matrix(train_data[, c("Lag1", "Lag2")])
dtrain <- xgb.DMatrix(data = features, label = label_column)
# Train the model
model <- xgboost(data = dtrain, params = params, nrounds = 100)

predictions <- predict(model, xgb.DMatrix(as.matrix(test_data[, c("Lag1", "Lag2")])))

# Evaluate the Model
RMSE(predictions,test_data$wti)
MAE(predictions,test_data$wti)
Rsquared(predictions,test_data$wti)


#plot predictions#
wtiB <- subset_df_xts$wti
wtiB <-na.omit(wtiB)
split_index <- floor(0.8 * nrow(wtiB))
wtiB$predict <- 0
wtiB$predict[(split_index + 1):nrow(wtiB)] <- predictions
wtidf <- as.data.frame(wtiB)
wtidf$date <- index(wtiB)
colnames(wtidf) <- c("Actual","prediction","Date")

merged <- wtidf[(split_index + 1):nrow(wtiB),]


df <- wtidf %>%
  gather(key = "Variable", value = "value", -Date) %>% 
  filter(value!=0)

tiff("xgboost_pred_wti.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g1xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

dev.off()


## for GAS####
#------------------------------------------******************

GPB <- subset_df_xts$GP
GPB <-na.omit(GPB)
GPdf <- as.data.frame(GPB)
GPdf$date <- index(GPB)
GPBoost <- GPdf %>%
  mutate(Lag1 = lag(GP),
         Lag2 = lag(GP, 2))
GPBoost <- na.omit(GPBoost)

train_size <- 0.8
train_index <- 1:(train_size * nrow(GPBoost))

train_data <- GPBoost[train_index, ]
test_data <- GPBoost[-train_index, ]



# Train the XGBoost Model
params <- list(
  objective = "reg:squarederror",  # Regression task
  eval_metric = "rmse"  # Root Mean Squared Error as the evaluation metric
)

label_column <- as.numeric(train_data$GP)
features <- as.matrix(train_data[, c("Lag1", "Lag2")])
library(xgboost)
dtrain <- xgb.DMatrix(data = features, label = label_column)

# Train the model
model <- xgboost(data = dtrain, params = params, nrounds = 100)

predictions_xgboost_gp <- predict(model, xgb.DMatrix(as.matrix(test_data[, c("Lag1", "Lag2")])))

# Evaluate the Model
RMSE(predictions_xgboost_gp,test_data$GP)
MAE(predictions_xgboost_gp,test_data$GP)
Rsquared(predictions_xgboost_gp,test_data$GP)

#plot predictions#
GPB <- subset_df_xts$GP
GPB <-na.omit(GPB)
split_index <- floor(0.8 * nrow(GPB))
GPB$predict <- 0
GPB$predict[(split_index + 1):nrow(GPB)] <- predictions_xgboost_gp
GPdf <- as.data.frame(GPB)
library(xts)
GPdf$date <- index(GPB)
colnames(GPdf) <- c("Actual","prediction","Date")


merged <- GPdf[(split_index + 1):nrow(GPB),]


df <- GPdf %>%
  gather(key = "Variable", value = "value", -Date) %>% 
  filter(value!=0)

tiff("xgboost_pred_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g2xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))
dev.off()





##for coal#####
#------------------------------------------******************
coalB <- subset_df_xts$coal
coalB <-na.omit(coalB)
coaldf <- as.data.frame(coalB)
library(xts)
coaldf$date <- index(coalB)
library(dplyr)
coalBoost <- coaldf %>%
  mutate(Lag1 = lag(coal),
         Lag2 = lag(coal, 2))
coalBoost <- na.omit(coalBoost)

train_size <- 0.8
train_index <- 1:(train_size * nrow(coalBoost))

train_data <- coalBoost[train_index, ]
test_data <- coalBoost[-train_index, ]



# Train the XGBoost Model
params <- list(
  objective = "reg:squarederror",  # Regression task
  eval_metric = "rmse"  # Root Mean Squared Error as the evaluation metric
)

label_column <- as.numeric(train_data$coal)
features <- as.matrix(train_data[, c("Lag1", "Lag2")])
library(xgboost)
dtrain <- xgb.DMatrix(data = features, label = label_column)

# Train the model
model <- xgboost(data = dtrain, params = params, nrounds = 100)

predictions <- predict(model, xgb.DMatrix(as.matrix(test_data[, c("Lag1", "Lag2")])))

# Evaluate the Model
RMSE(predictions,test_data$coal)
MAE(predictions,test_data$coal)
Rsquared(predictions,test_data$coal)

#plot predictions#
coalB <- subset_df_xts$coal
coalB <- na.omit(coalB)
split_index <- floor(0.8 * nrow(coalB)+1)
coalB$predict <- 0
coalB$predict[(split_index + 1):nrow(coalB)] <- predictions
coalB <-na.omit(coalB)
coaldf <- as.data.frame(coalB)
coaldf$date <- index(coalB)
colnames(coaldf) <- c("Actual","prediction","Date")


df <- coaldf %>%
  gather(key = "Variable", value = "value", -Date) %>% 
  filter(value!=0)

tiff("xgboost_pred_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g3xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))
dev.off()



# for 
bb <- coaldf[-train_index, ]






graphxgboost <- (g1xgboost+g2xgboost)/g3xgboost +plot_layout(guides="collect")

# Save the combined plot as a PNG file
ggsave("combined_plot_xgboost.png", width = 12, height = 6, plot = graphxgboost, dpi = 300)








#***********camparing accuracy difference***********************####


##Results1####
squaerror_wti_narx <- wtidf$error
squaerror_wti_ANN <- mergedwti$error
squaerror_wti_lstm <- combin$error
squaerror_wti_xgboost <- merged$error
##results 2#####
squaerror_gp_narx <- merged$error
squaerror_gp_ANN <- mergedgp$error
squaerror_gp_LSTM <- combin$error
squaerror_gp_xgboost <- merged$error
##results 3####
squaerror_coal_narx <- bb$error
squaerror_coal_ann <- bb$error
squaerror_coal_lstm <- combin$error
squaerror_coal_xgboost <- bb$error


library(forecast)
#repeat for each test 
dm.test(squaerror_coal_narx,squaerror_coal_xgboost,alternative = "less",
        h = 4, power = 2)








#*************** prediction in subsumples*********************####

subset_before <- subset_df_xts[index(subset_df_xts) <"2020-03-11"]
subset_after <- subset_df_xts[index(subset_df_xts) >= "2020-03-11"]

#******************NARX model before crisis******************####

# NARX model using neuralnet for WTI before crisis*####

exoge <-subset_before[,-1]
lag1_target <- lag(subset_before$wti, 1)
colnames(lag1_target) <- "wti1"
lag2_target <- lag(subset_before$wti, 2)
colnames(lag2_target) <- "wti2"
time <- length(subset_before$wti)
wti<-subset_before$wti
data <- data.frame(wti, exoge, lag1_target, lag2_target)
data <- data[complete.cases(data$wti), ] #delete missing value dor wti only
data[] <- lapply(data, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value

maxmindata <- as.data.frame(lapply(data, normalize))

## Split the data into training and testing sets
n<-length(maxmindata$wti)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#
## for all factors
formula <- wti ~ wti1 + wti2 + TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors

### Choose the best model##

grid <-  expand.grid(layer1 = c(1, 2,3),
                     layer2 = c(1, 2,3),
                     layer3 = c(1,2,3))
###with all variables##
library(dplyr)
library(caret)
subnn <- train(formula, 
            data = training_data, 
            method = "neuralnet", 
            tuneGrid = grid,
            metric = "RMSE",
            preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
            trControl = trainControl(
              method = "cv",
              number = 5,
              verboseIter = TRUE)
)

subnn
predictions <-  predict(subnn, newdata = testing_data)

# NARX model using neuralnet for GAS before crisis*####

exoge <-subset_before[,-2]
lag1_target <- lag(subset_before$GP, 1)
colnames(lag1_target) <- "gp1"
lag2_target <- lag(subset_before$GP, 2)
colnames(lag2_target) <- "gp2"
time <- length(subset_before$GP)
gp<-subset_before$GP
data <- data.frame(gp, exoge, lag1_target, lag2_target)
data <- data[complete.cases(data$GP), ] #delete missing value dor wti only
data[] <- lapply(data, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value

maxmindata <- as.data.frame(lapply(data, normalize))

## Split the data into training and testing sets
n<-length(maxmindata$GP)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#
## for all factors
formula <- GP ~ gp1 + gp2 + TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors

### Choose the best model##

grid <-  expand.grid(layer1 = c(1, 2,3),
                     layer2 = c(1, 2,3),
                     layer3 = c(1,2,3))
###with all variables##
library(dplyr)
library(caret)
subnngp <- train(formula, 
               data = training_data, 
               method = "neuralnet", 
               tuneGrid = grid,
               metric = "RMSE",
               preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
               trControl = trainControl(
                 method = "cv",
                 number = 5,
                 verboseIter = TRUE)
)

subnngp
predictionsgpsub <-  predict(subnngp, newdata = testing_data)


# NARX model using neuralnet for Coal before crisis*####
exoge <-subset_before[,-3]
lag1_target <- lag(subset_before$coal, 1)
colnames(lag1_target) <- "coal1"
lag2_target <- lag(subset_before$coal, 2)
colnames(lag2_target) <- "coal2"
time <- length(subset_before$coal)
coal<-subset_before$coal
data <- data.frame(coal, exoge, lag1_target, lag2_target)
data <- data[complete.cases(data$coal), ]
data[] <- lapply(data, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))# replace NA with mean value
maxmindata <- as.data.frame(lapply(data, normalize))


## Split the data into training and testing sets
n<-length(maxmindata$coal)
train_frac <- 0.8
train_idx <- 1:round(train_frac * n)
test_idx <- (round(train_frac * n) + 1):n
training_data <- maxmindata[train_idx,]
testing_data <- maxmindata[test_idx,]
## Define the NARX formula#
## for all factors
formula <- coal ~ coal1 + coal2 + TB+LTY+I+SRV+GOP+GIPIO+M2+IPI+UMP+EEPH+#macro factors
  TEMP+co2_per_capita+#envt factors
  solarpv+solarthermal+solarpvthermalhybrid+wind+hydropower+marineandtidal+bioenergy+geothermal+#technological factors
  bioenergyLCOE+geothermalLCOE+ offshorewindLCOE+solarphotovoltaicLCOE+concentratedsolarLCOE+
  hydropowerLCOE+onshorewindLCOE+AC_WIND_E+AC_SOLAR_E+AC_HYD_E+#renewable factors
  GR+EPI# political factors

### Choose the best model##

grid <-  expand.grid(layer1 = c(1, 2,3),
                     layer2 = c(1, 2,3),
                     layer3 = c(1,2,3))

###with all variables##
nncoalsub <- train(formula, 
                data = training_data, 
                method = "neuralnet", 
                tuneGrid = grid,
                metric = "RMSE",
                preProc = c("center", "scale", "nzv"), #good idea to do this with neural nets - your error is due to non scaled data
                trControl = trainControl(
                  method = "cv",
                  number = 5,
                  verboseIter = TRUE)
)

nncoalsub  # find the best model


# plot predictions and actual
predictions_coal_bef <-  predict(nncoalsub, newdata = testing_data)
data_xts <- as.xts(data)
maxmindata_xts <-lapply(data_xts, normalize)
coal_xts <- xts(maxmindata_xts$coal, order.by = index(maxmindata_xts[["coal"]]))
coal_xts$predictions <- 0
coal_xts$predictions[test_idx,] <- predictions_coal_bef
coaldf <- as.data.frame(coal_xts)
coaldf$Date <- index(coal_xts)
colnames(coaldf) <- c("Actual","prediction","Date")

df <- coaldf %>%
  gather(key = "Variable", value = "value", -Date) %>%
  filter(value!=0)

tiff("NARX_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), size = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))

dev.off()


#---------------AUTOML using h2o-------------------------------

library(h2o)
h2o.init()
library(writexl)
dataauto <- maxmindata[,-c(36:39)]
write_xlsx(dataauto, "dataauto.xlsx")
#convert to cvs
library(readxl)
files.to.read <- list.files(pattern = "dataauto.xlsx")
# Read each file and write it to CSV
lapply(files.to.read, function(f) {
  df <- read_excel(f, sheet = 1)
  write.csv(df, gsub("dataauto.xlsx", "dataauto.csv", f), row.names = FALSE)
})

dataauto <- h2o.importFile("dataauto.csv")
split <- h2o.splitFrame(dataauto, ratios = c(0.8))
train <- split[[1]]
test <- split[[2]]

# Define the predictor and response variables
x <- setdiff(names(dataauto[,-3]), c("response_variable"))
y <- "wti"



# # Train AutoML


automl_fit <- h2o.automl(
  x = x,
  y = y,
  training_frame = train,
  max_models = 10,  # Optional: Maximum number of models to evaluate
  max_runtime_secs = 3600  # Optional: Maximum training time in seconds
)

predictions <- h2o.predict(automl_fit, test)
print(predictions, n=100)
actual_values <- as.data.frame(test$wti)

predicted_values <- as.data.frame(predictions$predict)
comparison <- cbind(actual_values, predicted_values)

# Calculate evaluation metrics
accuracy <- mean(comparison$coal == comparison$predict)
mse <- mean((comparison$wti - comparison$predict)^2)
mae <- MAE1(comparison$wti,comparison$predict)
rmse <-RMSE(comparison$wti,comparison$predict)
squared <-Rsquared(comparison$wti,comparison$predict)


library(performance)
m <- lm(wti~.,maxmindata)
check_model(m)
check_normality(m)
check_autocorrelation(m)

#---------------AUTOML using autoML package-------------------------------
MAE1 <- function (true_values,predictions) {
  
  abs_diff <- abs((true_values - predictions))
  mad <- mean(abs_diff)
  cat("Mean Absolute Deviation (MAD):", mad, "\n") 
  
}
library(automl)
dataauto <- maxmindata[,-c(36:39)]
xmat <- dataauto[,-3]
ymat<- dataauto[,3]
amlmodel <- automl_train(Xref = xmat, Yref = ymat)
res <- cbind(ymat, automl_predict(model = amlmodel, X = xmat))
colnames(res) <- c('actual', 'predict')
res <- as.data.frame(res)

mae <- MAE1(res$actual,res$predict)
rmse <-RMSE(res$actual,res$predict)
squared <-Rsquared(res$actual,res$predict)

#----------------autoML using prophet-----------------------
library(prophet)
data <- subset_df_filled[,c(1:2)]
colnames(data)<- c("ds","y")
m <- prophet(data.frame(ds = data$ds, y = data$y), fit = FALSE)
m <- fit.prophet(m)

future <- make_future_dataframe(m, periods = 365, freq = "day")  # Adjust the number of periods as needed






#---------------AUTOML using autokeras package-------------------------------
# Load required libraries
library(autokeras)


# Load the dataset
data <- maxmindata[,-c(36,39)]
# Prepare the data


# 
# library(dplyr)
# data <- data %>%
#   arrange(Date) %>%
#   mutate(
#     Lag1 = lag(wti, 1),
#     Lag2 = lag(wti, 2),
#     Lag3 = lag(wti, 3)
#   ) %>%
#   na.omit()



# Split data into training and validation sets
split_index <- floor(0.8 * nrow(data))
train_data <- data[1:split_index, ]
test_data <- data[(split_index + 1):nrow(data),]


# wti will be the interest column to predict


train_file <- paste0(tempdir(),"/trainfile.csv")
write.csv(train_data, train_file, row.names = FALSE)

# file to predict,
test_file_to_predict <- paste0(tempdir(), "/testfile.csv")
write.csv(test_data[, -1], test_file_to_predict, row.names = FALSE)

test_file_to_eval <- paste0(tempdir(), "/fileevaluate.csv")
write.csv(test_data, test_file_to_eval, row.names = FALSE)

library(autokeras)
library(dplyr)
# Initialize the structured data regressor
reg <- model_structured_data_regressor(max_trials = 10) %>% # It tries 10 different models
  fit(train_file, "coal")

(predicted_y <- reg %>% predict(test_file_to_predict))
reg %>% evaluate(test_file_to_eval, "coal")
export_model(reg)











#---------------------------------END---------------------------------------------
