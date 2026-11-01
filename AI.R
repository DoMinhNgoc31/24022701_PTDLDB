#--------Title:  Predicting energy commodity prices amidst -------------
#-------worldwide energy transitions using deep learning models --
#
# Phiên bản đã cập nhật để chạy trên R >= 4.3 và RStudio / Positron mới.
# Các chỗ sửa đều có chú thích [UPDATED] (tương thích phiên bản) hoặc [FIX] (lỗi logic / biến chưa định nghĩa).
#   * keras (v2)  -> keras3        : LSTM viết lại theo Keras 3
#   * xgboost()   -> xgb.train()   : giao diện cũ (data=, params=) đã thay đổi từ xgboost 2.x/3.x
#   * ggplot2     : size -> linewidth ; legend.position dạng số -> "inside"
#   * tidyr       : gather() -> pivot_longer()
#   * dplyr::lag  : dùng stats::lag() cho đối tượng xts
#   * bỏ fdesc, mltsp, flexsurv, tensorflow, reticulate (không còn dùng / không có trên CRAN)
#   * dataset.xlsx : đường dẫn tương đối (cùng thư mục với AI.Rproj)
#
# LẦN ĐẦU CHẠY: sau khi cài các gói, chạy thêm MỘT LẦN:   keras3::install_keras()
#
##library#####
core_pkgs <- c("dplyr", "tidyr", "keras3", "neuralnet", "zoo", "caret",
               "NeuralNetTools", "readxl", "xts", "writexl", "xgboost",
               "ggplot2", "e1071", "forecast", "iml", "patchwork", "gridExtra")
# Gói tùy chọn, chỉ dùng cho phần AutoML ở cuối file: h2o, performance, see, automl, prophet
missing_pkgs <- setdiff(core_pkgs, rownames(installed.packages()))
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)

library(dplyr)
library(tidyr)
library(keras3)      # [UPDATED] thay cho keras (v2) + tensorflow
library(neuralnet)
library(zoo)
library(caret)
library(NeuralNetTools)
library(readxl)
library(xts)
library(writexl)
library(xgboost)
library(ggplot2)
library(e1071)
library(forecast)
library(patchwork)
index <- zoo::index
#************import data********************-----
# [UPDATED] đường dẫn tương đối: đặt dataset.xlsx cùng thư mục với AI.Rproj
# (RStudio tự đặt working directory về thư mục dự án khi mở file .Rproj)
excel_file <- "dataset.xlsx"
# excel_file <- "C:/Users/Foued Azuz 14/OneDrive/AI energy/dataset.xlsx"   # đường dẫn cũ
stopifnot("Không tìm thấy dataset.xlsx trong thư mục làm việc" = file.exists(excel_file))
sheet_names <- excel_sheets(excel_file)
all_dataframes <- lapply(sheet_names, function(sheet) {
    read_excel(excel_file, sheet = sheet)
  })

all_dataframes <- lapply(all_dataframes, as.data.frame)
  
for (i in seq_along(all_dataframes)) {   # [UPDATED] thay cho 1:22 cứng
    
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value")
ggplot(df, aes(x = Date, y = value, linetype = Variable)) + 
  geom_line(aes(color = Variable), linewidth = 1) +
  scale_color_manual(values = c("#59504E", "#59504E", "#222222")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Commodity", linetype = "Commodity")+
theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'))




#------------prepare the dataset for missing value--------------####
column_names <- colnames(merged_df)
start_date <- as.Date("2006-10-16")
end_date <- as.Date("2023-07-24")
mask <- merged_df$date >= start_date & merged_df$date <= end_date
subset_df <- merged_df[mask, ]
# [FIX] as.xts(data.frame) cần rownames là ngày -> tạo xts trực tiếp từ cột date
subset_df_xts <- xts(subset_df[, -1], order.by = subset_df$date)

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

# [FIX] as.xts(data.frame) cần rownames là ngày -> tạo xts trực tiếp từ cột date
subset_df_xts <- xts(subset_df[, -1], order.by = subset_df$date)

# [UPDATED] desc()/export() (gói fdesc) không còn khả dụng -> tự tính bảng thống kê mô tả
num_cols <- subset_df[, -1, drop = FALSE]
mytable <- data.frame(
  Variable = colnames(num_cols),
  N      = sapply(num_cols, function(x) sum(!is.na(x))),
  Mean   = sapply(num_cols, function(x) mean(x, na.rm = TRUE)),
  SD     = sapply(num_cols, function(x) sd(x, na.rm = TRUE)),
  Min    = sapply(num_cols, function(x) min(x, na.rm = TRUE)),
  Median = sapply(num_cols, function(x) median(x, na.rm = TRUE)),
  Max    = sapply(num_cols, function(x) max(x, na.rm = TRUE)),
  row.names = NULL
)
write_xlsx(mytable, "summary_statistics.xlsx")


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

# [FIX] MAE1 phải được định nghĩa trước khi dùng (phần h2o ở cuối file gọi MAE1 trước khi nó được tạo)
MAE1 <- function (true_values,predictions) {
  abs_diff <- abs((true_values - predictions))
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
# [FIX] nnn chưa định nghĩa -> dùng nnwti
plotnet(nnwti$finalModel,cex_val =0.5,"","WTI",bord_col="black",pad_x=0.75,
       bias=FALSE)
title("Oil wti")
dev.off()



#plot predictions vs actual
predictions <-  predict(nnwti, newdata = testing_data)   # [FIX] nnwti1 chưa định nghĩa

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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("predi_narx.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'))

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
print(g1grevrey)   # [FIX] vẽ ggplot vào file ảnh
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
nnGP <- caret::train(formula, 
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("NARX_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'))

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
print(g2grevrey)   # [FIX] vẽ ggplot vào file ảnh
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
nncoal <- caret::train(formula, 
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("predi_narx_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'))

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
print(g3grevrey)   # [FIX] vẽ ggplot vào file ảnh
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

nnwtimacro <- caret::train(formula, 
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

nngpmacro <- caret::train(formula, 
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

nncoalmacro <- caret::train(formula, 
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
  

ann <- caret::train(formula, 
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
predictions <- predict(ann, testing_data)   # [FIX] dùng mô hình ANN của WTI (trước đây gọi nhầm nngpmacro)

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
wtiB$predict[(split_index + 1):nrow(wtiB)] <- revpredict   # [FIX] sai thứ tự ưu tiên toán tử
wtidf <- as.data.frame(wtiB)
wtidf$date <- index(wtiB)
colnames(wtidf) <- c("Actual","prediction","Date")

df <- wtidf %>%
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("ANN_pred_wti.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g1 <-ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

print(g1)   # [FIX] vẽ ggplot vào file ảnh
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
ann1 <- caret::train(formula, 
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("ANN_pred_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g2 <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

print(g2)   # [FIX] vẽ ggplot vào file ảnh
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
ann2 <- caret::train(formula, 
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("ANN_pred_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g3 <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

print(g3)   # [FIX] vẽ ggplot vào file ảnh
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

# [UPDATED] Keras 3 (gói keras3): không còn tham số batch_input_shape cho LSTM stateful.
# Dùng functional API với keras_input(batch_shape = ...).
# Lần đầu chạy cần: keras3::install_keras()
inputs <- keras_input(batch_shape = c(batch_size, X_shape2, X_shape3))
lstm_layer <- layer_lstm(units = units, stateful = TRUE)
outputs <- lstm_layer(inputs) %>%
  layer_dense(units = 1)
model <- keras_model(inputs, outputs)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
  metrics = list("mae")
)
summary(model)

Epochs = 50
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  reset_state(lstm_layer)   # [UPDATED] đổi reset_states() -> reset_state() (keras3)
}

L = length(x_test)
scaler = Scaled$scaler
predictions = numeric(L)

for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  yhat = predict(model, X, batch_size = batch_size, verbose = 0)   # [UPDATED] keras3
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  # invert differencing
  yhat  = yhat + as.numeric(coal[(n+i)])
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

# tiff("LSTM_pred_gp.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution
library(ggplot2)
g1lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))




g2lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))


g3lstm <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
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
model <- xgb.train(params = params, data = dtrain, nrounds = 100)   # [UPDATED] xgboost() kiểu cũ đã bị thay đổi

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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>% 
  filter(value!=0)

tiff("xgboost_pred_wti.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g1xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "WTI")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))

print(g1xgboost)   # [FIX] vẽ ggplot vào file ảnh
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
model <- xgb.train(params = params, data = dtrain, nrounds = 100)   # [UPDATED] xgboost() kiểu cũ đã bị thay đổi

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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>% 
  filter(value!=0)

tiff("xgboost_pred_GP.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g2xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "GP")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))
print(g2xgboost)   # [FIX] vẽ ggplot vào file ảnh
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
model <- xgb.train(params = params, data = dtrain, nrounds = 100)   # [UPDATED] xgboost() kiểu cũ đã bị thay đổi

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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>% 
  filter(value!=0)

tiff("xgboost_pred_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

g3xgboost <- ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction", title = "Coal")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        plot.title = element_text(hjust = 0.5))
print(g3xgboost)   # [FIX] vẽ ggplot vào file ảnh
dev.off()



# for 
bb <- coaldf[-train_index, ]






graphxgboost <- (g1xgboost+g2xgboost)/g3xgboost +plot_layout(guides="collect")

# Save the combined plot as a PNG file
ggsave("combined_plot_xgboost.png", width = 12, height = 6, plot = graphxgboost, dpi = 300)








#***********camparing accuracy difference***********************####


# [FIX] Khối so sánh Diebold-Mariano dưới đây tham chiếu các cột không tồn tại
# (wtidf$error, mergedwti$error, combin$error, merged$error, bb$error...) nên không chạy được.
# Cần tính vectơ sai số của từng mô hình trên CÙNG tập test rồi mới dùng dm.test().
# ##Results1####
# squaerror_wti_narx <- wtidf$error
# squaerror_wti_ANN <- mergedwti$error
# squaerror_wti_lstm <- combin$error
# squaerror_wti_xgboost <- merged$error
# ##results 2#####
# squaerror_gp_narx <- merged$error
# squaerror_gp_ANN <- mergedgp$error
# squaerror_gp_LSTM <- combin$error
# squaerror_gp_xgboost <- merged$error
# ##results 3####
# squaerror_coal_narx <- bb$error
# squaerror_coal_ann <- bb$error
# squaerror_coal_lstm <- combin$error
# squaerror_coal_xgboost <- bb$error


# library(forecast)
# #repeat for each test 
# dm.test(squaerror_coal_narx,squaerror_coal_xgboost,alternative = "less",
#         h = 4, power = 2)



#*************** prediction in subsumples*********************####

subset_before <- subset_df_xts[index(subset_df_xts) <"2020-03-11"]
subset_after <- subset_df_xts[index(subset_df_xts) >= "2020-03-11"]

# [UPDATED] dplyr::lag() không tương thích với xts -> dùng stats::lag() (lag.xts) ở phần dưới
#******************NARX model before crisis******************####

# NARX model using neuralnet for WTI before crisis*####

exoge <-subset_before[,-1]
lag1_target <- stats::lag(subset_before$wti, 1)
colnames(lag1_target) <- "wti1"
lag2_target <- stats::lag(subset_before$wti, 2)
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
subnn <- caret::train(formula, 
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
lag1_target <- stats::lag(subset_before$GP, 1)
colnames(lag1_target) <- "gp1"
lag2_target <- stats::lag(subset_before$GP, 2)
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
subnngp <- caret::train(formula, 
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
lag1_target <- stats::lag(subset_before$coal, 1)
colnames(lag1_target) <- "coal1"
lag2_target <- stats::lag(subset_before$coal, 2)
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
nncoalsub <- caret::train(formula, 
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
  pivot_longer(-Date, names_to = "Variable", values_to = "value") %>%
  filter(value!=0)

tiff("NARX_coal.jpg",width = 10, height = 6, units = 'in', res = 350) #for high resolution

ggplot(df, aes(x = Date, y = value)) +
  geom_line(aes(color = Variable), linewidth = 1)+
  scale_color_manual(values = c("black", "red")) +
  theme(legend.position = "inside", legend.position.inside = c(0.1, 0.80))+
  labs(color = "Actual vs Prediction")+
  theme(panel.background = element_blank())+
  theme(axis.line.x = element_line(colour = 'black', linewidth=0.5, linetype='solid'),
        axis.line.y = element_line(colour = 'black', linewidth=0.5, linetype='solid'))

dev.off()


#---------------AUTOML using h2o-------------------------------

if (requireNamespace("h2o", quietly = TRUE)) {   # [UPDATED] phần tùy chọn
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
} else message("Bỏ qua h2o AutoML: chưa cài gói h2o")


if (requireNamespace("performance", quietly = TRUE) && requireNamespace("see", quietly = TRUE)) {
library(performance)
m <- lm(wti~.,maxmindata)
check_model(m)
check_normality(m)
check_autocorrelation(m)
} else message("Bỏ qua performance::check_model: cần cài gói performance và see")

#---------------AUTOML using autoML package-------------------------------
# [FIX] MAE1 đã được chuyển lên mục "define functions" (trước đây định nghĩa sau khi đã được gọi)
if (requireNamespace("automl", quietly = TRUE)) {   # [UPDATED] phần tùy chọn
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
} else message("Bỏ qua automl: chưa cài gói automl")

#----------------autoML using prophet-----------------------
if (requireNamespace("prophet", quietly = TRUE)) {   # [UPDATED] phần tùy chọn
library(prophet)
data <- subset_df[, c("date", "wti")]   # [FIX] subset_df_filled chưa định nghĩa -> subset_df (dữ liệu đã điền NA)
colnames(data)<- c("ds","y")
m <- prophet(data.frame(ds = data$ds, y = data$y))   # [FIX] fit.prophet() cần tham số df

future <- make_future_dataframe(m, periods = 365, freq = "day")  # Adjust the number of periods as needed
} else message("Bỏ qua prophet: chưa cài gói prophet")






#---------------AUTOML using autokeras package-------------------------------
# [UPDATED] Gói R "autokeras" (bản cuối 2021) phụ thuộc keras v2 / Keras 2, xung đột với keras3 -> TẮT mặc định.
# Đổi FALSE thành TRUE chỉ khi bạn tự dựng môi trường Python/keras cũ riêng cho phần này.
if (FALSE) {
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
}











#---------------------------------END---------------------------------------------



#==================================================================================================
#  >>> PHẦN MỞ RỘNG (EXTENDED EXPERIMENTS) – THÊM VÀO CUỐI FILE <<<
#
#  Phương pháp chọn từ Table 1 (Related Studies): SVR – Support Vector Regression
#  (Göncü, Kuzubaş & Saltoğlu 2024 dùng SVR cho giá dầu; Zhang et al. 2022 dùng SVR trong mô hình lai).
#  Lý do chọn: chạy bằng gói e1071 (đã nạp sẵn ở đầu file), không cần Python/keras, hợp với cỡ mẫu ~4.000 quan sát.
#
#  So sánh SVR với các mô hình của bài báo, trên CÙNG dữ liệu, CÙNG chia 80/20, CÙNG thang đo min-max [0,1]:
#     NARX (neuralnet + PCA) | XGBoost | Naive (random walk: y[t] = y[t-1])
#  (LSTM không chạy lại ở đây vì tốn thời gian và cần Python; số liệu LSTM của bài báo in ở bảng tham chiếu.)
#
#  Các thí nghiệm:
#     E1  So sánh chính trên tập test + kiểm định Diebold-Mariano (SVR vs từng mô hình)
#     E2  SVR: so sánh kernel (RBF / tuyến tính / đa thức bậc 2) và SVR + PCA
#     E3  SVR: loại bỏ nhóm biến (nhóm nào đóng góp cho dự báo? kiểm tra luận điểm "năng lượng tái tạo" của bài báo)
#     E4  Sai số theo giai đoạn trong tập test: COVID-19 vs chiến tranh Nga-Ukraine (từ 2022-02-24)
#     E5  Dự báo nhiều bước (h = 1, 5, 10, 20 ngày) – bài báo chỉ dự báo 1 bước
#
#  CÁCH CHẠY: chạy hết phần đầu file (nạp dữ liệu, định nghĩa hàm, tới dòng ~205) rồi chạy cả khối này.
#  Nếu đã chạy các phần NARX ở trên (nnwti, nnGP, nncoal) thì khối này dùng lại chúng; nếu chưa thì tự huấn luyện
#  lại NARX với cấu trúc nơron trong Figure 3: dầu (4,4,2), khí (4,4,4), than (4,3,2).
#  Chạy thử nhanh: đặt EXT_FAST <- TRUE (lưới siêu tham số nhỏ).  Kết quả ghi ra ext_SVR_extended_results.xlsx + 3 ảnh PNG.
#==================================================================================================

if (!exists("subset_df_xts") || !exists("normalize")) {
  stop("Chưa có 'subset_df_xts' / 'normalize'. Hãy chạy phần nạp dữ liệu và định nghĩa hàm (từ đầu file tới dòng ~205) trước.")
}
suppressPackageStartupMessages({
  library(e1071); library(caret); library(xgboost); library(forecast)
  library(ggplot2); library(dplyr); library(tidyr); library(writexl); library(zoo); library(xts)
})

#---- Cấu hình ----------------------------------------------------------------------------------
EXT_SEED       <- 2024
EXT_TRAIN_FRAC <- 0.8                   # như bài báo: 80% train / 20% test (~ mốc 11/03/2020)
EXT_N_LAGS     <- 4                     # 4 độ trễ của biến mục tiêu (như NARX)
EXT_HORIZONS   <- c(1, 5, 10, 20)       # E5: số ngày dự báo trước
EXT_WAR_DATE   <- as.Date("2022-02-24") # E4: mốc chia giai đoạn trong tập test
EXT_CV_FOLDS   <- 4                     # số fold rolling-origin khi tune SVR
EXT_FAST       <- FALSE                 # TRUE = lưới siêu tham số nhỏ, chạy nhanh để thử

EXT_TARGETS <- list(
  OIL  = list(target = "wti",  narx_obj = "nnwti",  hidden = c(4, 4, 2)),
  GAS  = list(target = "GP",   narx_obj = "nnGP",   hidden = c(4, 4, 4)),
  COAL = list(target = "coal", narx_obj = "nncoal", hidden = c(4, 3, 2))
)

# Nhóm biến: đúng như công thức NARX và danh sách 'groups' ở phần trên của file
EXT_GROUPS <- list(
  macro   = c("TB", "LTY", "I", "SRV", "GOP", "GIPIO", "M2", "IPI", "UMP", "EEPH"),
  env     = c("TEMP", "co2_per_capita"),
  tech    = c("solarpv", "solarthermal", "solarpvthermalhybrid", "wind",
              "hydropower", "marineandtidal", "bioenergy", "geothermal"),
  renew   = c("bioenergyLCOE", "geothermalLCOE", "offshorewindLCOE", "solarphotovoltaicLCOE",
              "concentratedsolarLCOE", "hydropowerLCOE", "onshorewindLCOE",
              "AC_WIND_E", "AC_SOLAR_E", "AC_HYD_E"),
  politic = c("GR", "EPI")
)

#---- Hàm 1: dựng dữ liệu y hệt các khối NARX ở trên (4 lag, bỏ 8 dòng đầu, min-max, chia 80/20) -------
ext_build <- function(target, xts_all = subset_df_xts) {
  stopifnot(target %in% colnames(xts_all))
  n_lags <- EXT_N_LAGS
  y_xts  <- xts_all[, target]
  exoge  <- xts_all[, setdiff(colnames(xts_all), target)]
  dat    <- data.frame(y_xts, exoge)
  keep   <- !is.na(dat[[target]])
  dat    <- dat[keep, , drop = FALSE]
  d_all  <- zoo::index(xts_all)[keep]
  y_obs  <- as.numeric(dat[[target]])

  lagm   <- embed(y_obs, n_lags + 1)[-seq_len(n_lags), , drop = FALSE]  # cột 1 = y[t], cột 2.. = lag 1..4
  drop_n <- 2 * n_lags                                                  # = 8 dòng đầu như code gốc
  dat    <- dat[-seq_len(drop_n), , drop = FALSE]
  d_all  <- d_all[-seq_len(drop_n)]
  stopifnot(nrow(lagm) == nrow(dat),
            isTRUE(all.equal(unname(lagm[, 1]), unname(dat[[target]]))))  # kiểm tra căn hàng

  lag_names <- paste0(tolower(target), seq_len(n_lags))                 # wti1.., gp1.., coal1.. (khớp nnwti/nnGP/nncoal)
  lag_df    <- as.data.frame(lagm[, -1, drop = FALSE])
  names(lag_df) <- lag_names
  dat <- cbind(dat, lag_df)

  dat[] <- lapply(dat, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))  # NA -> trung bình (như code gốc)
  md    <- as.data.frame(lapply(dat, normalize))                                 # min-max như bài báo
  md[]  <- lapply(md, function(x) { x[!is.finite(x)] <- 0; x })

  feats_all <- c(lag_names, unlist(EXT_GROUPS, use.names = FALSE))
  miss <- setdiff(feats_all, names(md))
  if (length(miss) > 0) stop("Thiếu biến trong dữ liệu: ", paste(miss, collapse = ", "))

  n    <- nrow(md)
  n_tr <- round(EXT_TRAIN_FRAC * n)
  list(target = target, md = md, dates = d_all, n = n, n_train = n_tr,
       train_idx = seq_len(n_tr), test_idx = (n_tr + 1):n,
       lag_names = lag_names, feats_all = feats_all,
       range_y = diff(range(dat[[target]])))                              # để quy đổi RMSE về đơn vị giá (USD)
}

#---- Hàm 2: chỉ số đánh giá --------------------------------------------------------------------------
ext_metrics <- function(actual, pred, prev, range_y = 1, naive_rmse = NA_real_, is_naive = FALSE) {
  ok <- is.finite(actual) & is.finite(pred) & is.finite(prev)
  if (sum(ok) < 10) {
    return(data.frame(RMSE = NA_real_, MAE = NA_real_, R2 = NA_real_, DA = NA_real_,
                      RMSE_vs_Naive = NA_real_, RMSE_orig = NA_real_, MAE_orig = NA_real_))
  }
  a <- actual[ok]; p <- pred[ok]; pv <- prev[ok]
  rmse <- sqrt(mean((a - p)^2))
  mae  <- mean(abs(a - p))
  r2   <- suppressWarnings(stats::cor(a, p)^2)                 # R2 = bình phương tương quan (phương trình 19 của bài báo)
  chg  <- a != pv
  da   <- if (is_naive || sum(chg) == 0) NA_real_ else mean(sign(p[chg] - pv[chg]) == sign(a[chg] - pv[chg]))
  data.frame(RMSE = rmse, MAE = mae, R2 = r2, DA = da,
             RMSE_vs_Naive = rmse / naive_rmse,                 # < 1: tốt hơn Naive
             RMSE_orig = rmse * range_y, MAE_orig = mae * range_y)
}

#---- Hàm 3: SVR (e1071) ------------------------------------------------------------------------------
ext_grid <- function(kernel) {
  if (EXT_FAST) {
    return(switch(kernel,
      linear     = expand.grid(cost = 1,       gmult = 1,        epsilon = 0.1),
      polynomial = expand.grid(cost = c(1, 10), gmult = 0.25,    epsilon = 0.1),
                   expand.grid(cost = c(1, 10), gmult = c(0.25, 1), epsilon = 0.1)))
  }
  switch(kernel,
    linear     = expand.grid(cost = c(0.1, 1, 10),  gmult = 1,              epsilon = c(0.01, 0.1)),
    polynomial = expand.grid(cost = c(0.1, 1, 10),  gmult = c(0.25, 1),     epsilon = c(0.01, 0.1)),
                 expand.grid(cost = c(1, 10, 100),  gmult = c(0.25, 1, 4),  epsilon = c(0.01, 0.1)))
}

ext_svr_fit <- function(X, y, kernel, cost, gmult, epsilon) {
  args <- list(x = X, y = y, type = "eps-regression", kernel = kernel,
               cost = cost, gamma = gmult / ncol(X), epsilon = epsilon, scale = TRUE)
  if (kernel == "polynomial") { args$degree <- 2; args$coef0 <- 1 }
  suppressWarnings(do.call(e1071::svm, args))
}

# Tune bằng rolling-origin CV (cửa sổ mở rộng, không xáo trộn thời gian -> không rò rỉ tương lai)
ext_svr_tune <- function(X, y, kernel = "radial") {
  grid <- ext_grid(kernel)
  n <- nrow(X); k <- EXT_CV_FOLDS
  cuts <- floor(seq(floor(0.4 * n), n, length.out = k + 1))
  grid$cv_rmse <- NA_real_
  for (g in seq_len(nrow(grid))) {
    errs <- rep(NA_real_, k)
    for (f in seq_len(k)) {
      tr <- seq_len(cuts[f])
      va <- (cuts[f] + 1):cuts[f + 1]
      m <- try(ext_svr_fit(X[tr, , drop = FALSE], y[tr], kernel,
                           grid$cost[g], grid$gmult[g], grid$epsilon[g]), silent = TRUE)
      if (inherits(m, "try-error")) next
      p <- as.numeric(predict(m, X[va, , drop = FALSE]))
      errs[f] <- sqrt(mean((y[va] - p)^2))
    }
    grid$cv_rmse[g] <- mean(errs, na.rm = TRUE)
  }
  if (all(!is.finite(grid$cv_rmse))) return(grid[1, , drop = FALSE])
  grid[which.min(grid$cv_rmse), , drop = FALSE]
}

# Bỏ biến gần như hằng số (nzv) dựa trên tập train; trả về ma trận train/test
ext_select_X <- function(md, feats, tr, te) {
  X  <- as.matrix(md[, feats, drop = FALSE])
  nz <- caret::nearZeroVar(X[tr, , drop = FALSE])
  if (length(nz) > 0) X <- X[, -nz, drop = FALSE]
  list(Xtr = X[tr, , drop = FALSE], Xte = X[te, , drop = FALSE])
}

ext_pca <- function(Xtr, Xte, thresh = 0.95) {
  pp <- caret::preProcess(as.data.frame(Xtr), method = c("center", "scale", "pca"), thresh = thresh)
  list(Xtr = as.matrix(predict(pp, as.data.frame(Xtr))),
       Xte = as.matrix(predict(pp, as.data.frame(Xte))))
}

#---- Hàm 4: XGBoost (cùng cách gọi xgb.train như phần xgboost ở trên) ------------------------------------
ext_xgb <- function(Xtr, ytr, Xte, tuned = TRUE) {
  set.seed(EXT_SEED)
  if (tuned) {
    params <- list(objective = "reg:squarederror", eval_metric = "rmse", eta = 0.05,
                   max_depth = 4, subsample = 0.8, colsample_bytree = 0.8)
    nr <- 300
  } else {                                   # giống code gốc của bài báo (tham số mặc định, 100 vòng)
    params <- list(objective = "reg:squarederror", eval_metric = "rmse")
    nr <- 100
  }
  dtr <- xgboost::xgb.DMatrix(data = as.matrix(Xtr), label = as.numeric(ytr))
  m   <- xgboost::xgb.train(params = params, data = dtr, nrounds = nr, verbose = 0)
  as.numeric(predict(m, xgboost::xgb.DMatrix(as.matrix(Xte))))
}

#---- Hàm 5: NARX của bài báo (dùng lại nnwti/nnGP/nncoal nếu có, nếu không thì huấn luyện lại) ------------
ext_narx <- function(ds, narx_obj, hidden) {
  te <- ds$md[ds$test_idx, , drop = FALSE]
  if (exists(narx_obj, envir = .GlobalEnv, inherits = FALSE)) {
    mdl <- get(narx_obj, envir = .GlobalEnv)
    p <- try(as.numeric(predict(mdl, newdata = te)), silent = TRUE)
    if (!inherits(p, "try-error") && length(p) == nrow(te) && all(is.finite(p))) {
      message("   NARX: dùng lại mô hình '", narx_obj, "' đã huấn luyện ở phần trên của file")
      return(p)
    }
  }
  message("   NARX: huấn luyện lại, cấu trúc ", paste(hidden, collapse = "-"), " (Figure 3 của bài báo)")
  form <- stats::as.formula(paste(ds$target, "~", paste(ds$feats_all, collapse = " + ")))
  set.seed(EXT_SEED)
  mdl <- try(suppressWarnings(caret::train(
    form, data = ds$md[ds$train_idx, , drop = FALSE], method = "neuralnet",
    tuneGrid = data.frame(layer1 = hidden[1], layer2 = hidden[2], layer3 = hidden[3]),
    metric = "RMSE", preProc = c("center", "scale", "nzv", "pca"),
    trControl = caret::trainControl(method = "none"))), silent = TRUE)
  if (inherits(mdl, "try-error")) {
    message("   NARX: huấn luyện thất bại (neuralnet không hội tụ) -> bỏ qua NARX cho biến này")
    return(rep(NA_real_, length(ds$test_idx)))
  }
  p <- try(as.numeric(predict(mdl, newdata = te)), silent = TRUE)
  if (inherits(p, "try-error")) return(rep(NA_real_, length(ds$test_idx)))
  p
}

#---- Hàm 6: các tập biến cho E3 và dữ liệu cho E5 ------------------------------------------------------
ext_feature_sets <- function(ds) {
  L <- ds$lag_names; G <- EXT_GROUPS
  list(
    "A0 Exogenous only (no lags)"              = unlist(G, use.names = FALSE),
    "A1 Lags only (AR)"                        = L,
    "A2 Lags + Macro"                          = c(L, G$macro),
    "A3 Lags + Macro + Env + Political"        = c(L, G$macro, G$env, G$politic),
    "A4 Lags + Tech + Renewable (transition)"  = c(L, G$tech, G$renew),
    "A5 All minus Renewable"                   = c(L, G$macro, G$env, G$politic, G$tech),
    "A6 All minus Technological"               = c(L, G$macro, G$env, G$politic, G$renew),
    "A7 All factors (full model)"              = ds$feats_all
  )
}

# Dự báo trực tiếp h bước: dùng thông tin tại thời điểm t (y[t], 4 lag, biến ngoại sinh) để dự báo y[t+h]
ext_horizon_xy <- function(ds, h) {
  y0 <- ds$md[[ds$target]]
  X  <- cbind(as.matrix(ds$md[, ds$feats_all, drop = FALSE]), y_t = y0)
  origin <- seq_len(ds$n - h)
  list(X = X[origin, , drop = FALSE], y = y0[origin + h], prev = y0[origin],
       tr = seq_len(ds$n_train - h),
       te = (ds$n_train - h + 1):(ds$n - h))      # mục tiêu của tập test luôn là các ngày (n_train+1):n
}

#---- Chạy toàn bộ thí nghiệm cho 1 biến (dầu / khí / than) ------------------------------------------------
ext_run_one <- function(label, cfg) {
  message("\n=== ", label, " (", cfg$target, ") ===")
  ds <- ext_build(cfg$target)
  tr <- ds$train_idx; te <- ds$test_idx
  y  <- ds$md[[ds$target]]
  y_te    <- y[te]
  prev_te <- ds$md[[ds$lag_names[1]]][te]                    # y[t-1]
  message(sprintf("   n = %d | train: %s -> %s | test: %s -> %s",
                  ds$n, format(ds$dates[tr[1]]), format(ds$dates[tail(tr, 1)]),
                  format(ds$dates[te[1]]), format(ds$dates[tail(te, 1)])))

  # ---- SVR: tune + 3 kernel + SVR/PCA
  sel <- ext_select_X(ds$md, ds$feats_all, tr, te)
  best <- list(); pred <- list()
  for (kern in c("radial", "linear", "polynomial")) {
    message("   SVR kernel = ", kern, ": tune bằng rolling-origin CV ...")
    bp <- ext_svr_tune(sel$Xtr, y[tr], kern)
    best[[kern]] <- bp
    m <- ext_svr_fit(sel$Xtr, y[tr], kern, bp$cost, bp$gmult, bp$epsilon)
    pred[[kern]] <- as.numeric(predict(m, sel$Xte))
  }
  message("   SVR kernel = radial + PCA ...")
  pc <- ext_pca(sel$Xtr, sel$Xte)
  bp <- ext_svr_tune(pc$Xtr, y[tr], "radial")
  best[["radial_pca"]] <- bp
  m  <- ext_svr_fit(pc$Xtr, y[tr], "radial", bp$cost, bp$gmult, bp$epsilon)
  pred[["radial_pca"]] <- as.numeric(predict(m, pc$Xte))

  # ---- Các mô hình đối chứng
  message("   XGBoost + NARX + Naive ...")
  xl        <- as.matrix(ds$md[, ds$lag_names[1:2], drop = FALSE])    # XGBoost của bài báo chỉ dùng Lag1, Lag2
  p_xgb_lag <- ext_xgb(xl[tr, , drop = FALSE], y[tr], xl[te, , drop = FALSE], tuned = FALSE)
  p_xgb_all <- ext_xgb(sel$Xtr, y[tr], sel$Xte, tuned = TRUE)
  p_narx    <- ext_narx(ds, cfg$narx_obj, cfg$hidden)

  preds <- data.frame(date = ds$dates[te], actual = y_te, prev = prev_te, check.names = FALSE)
  preds[["SVR-RBF"]]       <- pred[["radial"]]
  preds[["SVR-Linear"]]    <- pred[["linear"]]
  preds[["SVR-Poly2"]]     <- pred[["polynomial"]]
  preds[["SVR-RBF+PCA"]]   <- pred[["radial_pca"]]
  preds[["NARX (paper)"]]  <- p_narx
  preds[["XGBoost (lags)"]] <- p_xgb_lag
  preds[["XGBoost (all)"]] <- p_xgb_all
  preds[["Naive"]]         <- prev_te
  model_cols <- setdiff(names(preds), c("date", "actual", "prev"))

  # ---- E1: bảng chỉ số chính
  nv <- sqrt(mean((y_te - prev_te)^2))
  tab_main <- do.call(rbind, lapply(model_cols, function(mc) {
    data.frame(Commodity = label, Model = mc,
               ext_metrics(preds$actual, preds[[mc]], preds$prev, ds$range_y, nv, is_naive = (mc == "Naive")),
               check.names = FALSE)
  }))

  # ---- E1: Diebold-Mariano, SVR-RBF (phương pháp 1) vs mô hình khác (phương pháp 2)
  dm <- do.call(rbind, lapply(setdiff(model_cols, "SVR-RBF"), function(mc) {
    e1 <- preds$actual - preds[["SVR-RBF"]]
    e2 <- preds$actual - preds[[mc]]
    ok <- is.finite(e1) & is.finite(e2)
    r2s <- try(forecast::dm.test(e1[ok], e2[ok], alternative = "two.sided", h = 1, power = 2), silent = TRUE)
    rls <- try(forecast::dm.test(e1[ok], e2[ok], alternative = "less",      h = 1, power = 2), silent = TRUE)
    data.frame(Commodity = label, SVR = "SVR-RBF", vs = mc,
               DM_stat = if (inherits(r2s, "try-error")) NA_real_ else unname(as.numeric(r2s$statistic)),
               p_two_sided = if (inherits(r2s, "try-error")) NA_real_ else as.numeric(r2s$p.value),
               p_SVR_more_accurate = if (inherits(rls, "try-error")) NA_real_ else as.numeric(rls$p.value))
  }))

  # ---- E4: theo giai đoạn trong tập test
  per <- ifelse(preds$date < EXT_WAR_DATE, "1 Test start - 2022-02-23 (COVID-19 era)", "2 From 2022-02-24 (Russia-Ukraine war)")
  tab_sub <- do.call(rbind, lapply(sort(unique(per)), function(pp) {
    ii   <- which(per == pp)
    nv_p <- sqrt(mean((preds$actual[ii] - preds$prev[ii])^2))
    do.call(rbind, lapply(model_cols, function(mc) {
      data.frame(Commodity = label, Period = pp, n_days = length(ii), Model = mc,
                 ext_metrics(preds$actual[ii], preds[[mc]][ii], preds$prev[ii], ds$range_y, nv_p,
                             is_naive = (mc == "Naive")),
                 check.names = FALSE)
    }))
  }))

  # ---- E3: loại bỏ nhóm biến (dùng siêu tham số tốt nhất của SVR-RBF đầy đủ biến)
  message("   E3: loại bỏ nhóm biến ...")
  bpr <- best[["radial"]]
  fs  <- ext_feature_sets(ds)
  tab_abl <- do.call(rbind, lapply(names(fs), function(nm) {
    s <- ext_select_X(ds$md, fs[[nm]], tr, te)
    m <- ext_svr_fit(s$Xtr, y[tr], "radial", bpr$cost, bpr$gmult, bpr$epsilon)
    p <- as.numeric(predict(m, s$Xte))
    data.frame(Commodity = label, Feature_set = nm, n_features = ncol(s$Xtr),
               ext_metrics(y_te, p, prev_te, ds$range_y, nv), check.names = FALSE)
  }))

  # ---- E5: dự báo nhiều bước
  message("   E5: dự báo nhiều bước ...")
  tab_h <- do.call(rbind, lapply(EXT_HORIZONS, function(h) {
    hd <- ext_horizon_xy(ds, h)
    nz <- caret::nearZeroVar(hd$X[hd$tr, , drop = FALSE])
    Xh <- if (length(nz) > 0) hd$X[, -nz, drop = FALSE] else hd$X
    m     <- ext_svr_fit(Xh[hd$tr, , drop = FALSE], hd$y[hd$tr], "radial", bpr$cost, bpr$gmult, bpr$epsilon)
    p_svr <- as.numeric(predict(m, Xh[hd$te, , drop = FALSE]))
    p_xgb <- ext_xgb(Xh[hd$tr, , drop = FALSE], hd$y[hd$tr], Xh[hd$te, , drop = FALSE], tuned = TRUE)
    p_nv  <- hd$prev[hd$te]
    a     <- hd$y[hd$te]
    nv_h  <- sqrt(mean((a - p_nv)^2))
    mk <- function(nm, p, isn = FALSE) {
      data.frame(Commodity = label, Horizon_days = h, Model = nm,
                 ext_metrics(a, p, p_nv, ds$range_y, nv_h, is_naive = isn), check.names = FALSE)
    }
    out <- rbind(mk("SVR-RBF", p_svr), mk("XGBoost (all)", p_xgb), mk("Naive", p_nv, TRUE))
    out$DM_p_vs_Naive <- NA_real_
    r <- try(forecast::dm.test(a - p_svr, a - p_nv, alternative = "two.sided", h = h, power = 2), silent = TRUE)
    if (!inherits(r, "try-error")) out$DM_p_vs_Naive[out$Model == "SVR-RBF"] <- as.numeric(r$p.value)
    out
  }))

  tab_par <- do.call(rbind, lapply(names(best), function(k) {
    data.frame(Commodity = label, Kernel = k, best[[k]][, c("cost", "gmult", "epsilon", "cv_rmse")], row.names = NULL)
  }))

  list(preds = preds, tab_main = tab_main, dm = dm, tab_sub = tab_sub,
       tab_abl = tab_abl, tab_h = tab_h, tab_par = tab_par)
}

#---- CHẠY -------------------------------------------------------------------------------------------
ext_t0  <- Sys.time()
ext_res <- lapply(names(EXT_TARGETS), function(nm) ext_run_one(nm, EXT_TARGETS[[nm]]))
names(ext_res) <- names(EXT_TARGETS)

ext_bind <- function(field) {
  d <- do.call(rbind, lapply(ext_res, function(r) r[[field]]))
  rownames(d) <- NULL
  d
}
ext_tab_main <- ext_bind("tab_main")
ext_tab_dm   <- ext_bind("dm")
ext_tab_sub  <- ext_bind("tab_sub")
ext_tab_abl  <- ext_bind("tab_abl")
ext_tab_h    <- ext_bind("tab_h")
ext_tab_par  <- ext_bind("tab_par")
ext_tab_pred <- do.call(rbind, lapply(names(ext_res), function(nm) {
  data.frame(Commodity = nm, ext_res[[nm]]$preds, check.names = FALSE)
}))
rownames(ext_tab_pred) <- NULL

# Số liệu NARX / LSTM do bài báo công bố (Table 3) – chỉ để tham chiếu
ext_paper_ref <- data.frame(
  Commodity = rep(c("OIL", "GAS", "COAL"), 2),
  Model     = rep(c("NARX (paper, Table 3)", "LSTM (paper, Table 3)"), each = 3),
  MAE  = c(0.0086, 0.0070, 0.0033, 0.0058, 0.0098, 0.0020),
  RMSE = c(0.0117, 0.0112, 0.0057, 0.647,  0.0886, 0.569),
  R2   = c(0.9911, 0.9835, 0.9915, 0.9993, 0.9986, 0.9990)
)

ext_round <- function(d, k = 4) { d[] <- lapply(d, function(x) if (is.numeric(x)) round(x, k) else x); d }

cat("\n\n================ E1. SO SÁNH CHÍNH TRÊN TẬP TEST (thang min-max [0,1]) ================\n")
cat("RMSE_vs_Naive < 1 nghĩa là tốt hơn dự báo 'giá hôm qua'. DA = tỷ lệ đoán đúng hướng tăng/giảm.\n")
print(ext_round(ext_tab_main), row.names = FALSE)
cat("\n--- Diebold-Mariano: SVR-RBF so với mô hình khác (h = 1, bình phương sai số) ---\n")
cat("DM_stat < 0: SVR có sai số nhỏ hơn. p_SVR_more_accurate < 0.05: SVR chính xác hơn có ý nghĩa thống kê.\n")
print(ext_round(ext_tab_dm), row.names = FALSE)
cat("\n--- Tham chiếu: số liệu công bố trong bài báo (Table 3) ---\n")
print(ext_paper_ref, row.names = FALSE)
cat("\n================ E2. SIÊU THAM SỐ SVR ĐƯỢC CHỌN (gamma = gmult / số biến) ================\n")
print(ext_round(ext_tab_par), row.names = FALSE)
cat("\n================ E3. LOẠI BỎ NHÓM BIẾN (SVR-RBF) ================\n")
print(ext_round(ext_tab_abl), row.names = FALSE)
cat("\n================ E4. THEO GIAI ĐOẠN TRONG TẬP TEST ================\n")
print(ext_round(ext_tab_sub), row.names = FALSE)
cat("\n================ E5. DỰ BÁO NHIỀU BƯỚC (h ngày) ================\n")
print(ext_round(ext_tab_h), row.names = FALSE)

ext_cmp <- ext_tab_main %>%
  dplyr::filter(Model %in% c("SVR-RBF", "NARX (paper)")) %>%
  dplyr::select(Commodity, Model, RMSE) %>%
  tidyr::pivot_wider(names_from = Model, values_from = RMSE) %>%
  dplyr::mutate(`SVR vs NARX: chênh lệch RMSE (%)` = 100 * (`SVR-RBF` / `NARX (paper)` - 1))
cat("\n--- Tóm tắt: RMSE của SVR so với NARX (âm = SVR tốt hơn) ---\n")
print(ext_round(as.data.frame(ext_cmp)), row.names = FALSE)

ext_best <- ext_tab_main %>%
  dplyr::filter(Model != "Naive") %>%
  dplyr::group_by(Commodity) %>%
  dplyr::slice_min(RMSE, n = 1, with_ties = FALSE) %>%
  dplyr::ungroup() %>%
  dplyr::select(Commodity, Model, RMSE, MAE, R2)
cat("\n--- Mô hình có RMSE thấp nhất (không tính Naive) ---\n")
print(ext_round(as.data.frame(ext_best)), row.names = FALSE)

#---- Hình -------------------------------------------------------------------------------------------
ext_plot_df <- ext_tab_pred[, c("Commodity", "date", "actual", "SVR-RBF", "NARX (paper)")]
names(ext_plot_df) <- c("Commodity", "date", "Actual", "SVR-RBF", "NARX (paper)")
ext_plot_long <- tidyr::pivot_longer(ext_plot_df, cols = c("Actual", "SVR-RBF", "NARX (paper)"),
                                     names_to = "Series", values_to = "value")
ext_g1 <- ggplot(ext_plot_long, aes(x = date, y = value, color = Series)) +
  geom_line(linewidth = 0.5, na.rm = TRUE) +
  facet_wrap(~ Commodity, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c("Actual" = "black", "SVR-RBF" = "red", "NARX (paper)" = "blue")) +
  labs(title = "Tập test: giá thực tế vs dự báo (thang min-max)", x = "Date", y = "Normalized price", color = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
print(ext_g1)
ggsave("ext_fig1_forecast_SVR_vs_NARX.png", ext_g1, width = 10, height = 9, dpi = 300)

ext_tab_main$Model <- factor(ext_tab_main$Model, levels = unique(ext_tab_main$Model))
ext_g2 <- ggplot(ext_tab_main, aes(x = Model, y = RMSE, fill = grepl("SVR", Model))) +
  geom_col(na.rm = TRUE) +
  facet_wrap(~ Commodity, scales = "free_y") +
  scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "grey60"), guide = "none") +
  labs(title = "RMSE trên tập test (thấp hơn = tốt hơn)", x = NULL, y = "RMSE (min-max scale)") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 40, hjust = 1))
print(ext_g2)
ggsave("ext_fig2_RMSE_comparison.png", ext_g2, width = 12, height = 5, dpi = 300)

ext_g3 <- ggplot(dplyr::filter(ext_tab_h, Model != "Naive"),
                 aes(x = Horizon_days, y = RMSE_vs_Naive, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(na.rm = TRUE) + geom_point(na.rm = TRUE) +
  facet_wrap(~ Commodity, scales = "free_y") +
  labs(title = "Dự báo nhiều bước: RMSE / RMSE của Naive (dưới 1 = tốt hơn Naive)",
       x = "Số ngày dự báo trước (h)", y = "RMSE tương đối so với Naive", color = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
print(ext_g3)
ggsave("ext_fig3_multi_horizon.png", ext_g3, width = 10, height = 4.5, dpi = 300)

#---- Xuất Excel -------------------------------------------------------------------------------------
ext_tab_main$Model <- as.character(ext_tab_main$Model)
write_xlsx(list(E1_main = ext_tab_main, E1_DM_test = ext_tab_dm, E2_params = ext_tab_par,
                E3_feature_sets = ext_tab_abl, E4_subperiods = ext_tab_sub, E5_horizons = ext_tab_h,
                predictions = ext_tab_pred, paper_Table3_ref = ext_paper_ref),
           "ext_SVR_extended_results.xlsx")

cat("\nXong. Thời gian chạy: ", format(round(difftime(Sys.time(), ext_t0, units = "mins"), 1)), "\n",
    "File: ext_SVR_extended_results.xlsx, ext_fig1_forecast_SVR_vs_NARX.png, ",
    "ext_fig2_RMSE_comparison.png, ext_fig3_multi_horizon.png\n", sep = "")
#---------------------------------END OF EXTENDED EXPERIMENTS-------------------------------------
