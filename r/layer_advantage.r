library(ggplot2)
library(dplyr)
library(forcats)
library(stringr)
library(tikzDevice)
library(ggh4x)

options(tikzLatexPackages = c(getOption("tikzLatexPackages"),
                              "\\usepackage{amsmath}"))

source("r/base.r")

data <- read.csv("csvs/layer_advantage.csv", stringsAsFactors = FALSE)

offset <- 0
# dimension "layer" causes internal bug when calling facet_nested
data$layer2 <- data$layer
data$relative_performance <- data$relative_performance - offset


# data$depolarizing_noise <- factor(data$depolarizing_noise)
# data$thermal_relaxation_noise <- factor(data$thermal_relaxation_noise)

# data <- data %>% mutate(algorithm = fct_relevel(algorithm, 
#     "QAOA",
#     "WSQAOA",
#     "WS-Init-QAOA",
#     "RQAOA"
# ))
# facet_titles <- c('QAOA.Depth'="$p$",'Number.of.Qubits'="$n$")
facet_labeller <- function(variable, value) {
    return(paste(facet_titles[variable], value, sep=": "))
}

map_layer <- function(layer) {
    last_char <- str_sub(layer, -1, -1)
    if (last_char == "1") {
        return(str_c(layer, "st layer"))
    }
    if (last_char == "2") {
        return(str_c(layer, "nd layer"))
    }
    if (last_char == "3") {
        return(str_c(layer, "rd layer"))
    }
    return(str_c(layer, "th layer"))
}

label_layer <- function(layer) {
    return(sapply(layer, map_layer))
}

g <- ggplot(data=data,mapping=aes(x=depolarizing_noise, y=thermal_relaxation_noise, fill=relative_performance)) +
    # geom_line() +
    geom_tile() +
    # coord_equal() +
    labs(y="Thermal relaxation noise", x="Depolarising noise") +
    scale_x_continuous(breaks=seq(0, 1, 0.5)) +
    scale_y_continuous(breaks=seq(0, 1, 0.5)) +
    scale_fill_gradient2(
        name="Relative layer advantage",
        low="#ed665a",
        mid="white",
        high="#1f78b4",
        midpoint=1 - offset,
        breaks=seq(0.9 - offset, 1.1 - offset, 0.05)
    ) +
    # scale_fill_gradient2(low="white", mid="red", high="blue", midpoint=1) +
    facet_nested(algorithm ~ problem + layer2, labeller=labeller(
        problem=label_value, algorithm=label_value,
        layer2=label_layer
    )) +
    theme_paper_base() + theme(
        # legend.position="right",
        legend.title = element_text(size=BASE.SIZE),
        legend.text=element_text(size=6),
        legend.key.height=unit(0.3, "cm"),
        strip.text.y = element_text(size=4.5)
    )

save_name <- "layer_advantage"

WIDTH <- 1 * COLWIDTH
HEIGHT <- 0.85 * COLWIDTH
pdf(str_c(PDF_OUTDIR, save_name, ".pdf"), width = WIDTH, height = HEIGHT)
print(g)
dev.off()
tikz(str_c(TIKZ_OUTDIR, save_name, ".tex"), width = WIDTH, height = HEIGHT)
print(g)
dev.off()
