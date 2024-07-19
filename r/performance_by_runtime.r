library(ggplot2)
library(stringr)
library(forcats)
library(dplyr)
library(tikzDevice)
library(ggnewscale)
library(ggh4x)

options(tikzLatexPackages = c(getOption("tikzLatexPackages"),
                              "\\usepackage{amsmath}"))

source("r/base.r")

light_colors <- c("#333333", "#E9C77A")
data <- read.csv("csvs/performance_by_runtime.csv", stringsAsFactors = FALSE)
# average_data <- read.csv("csvs/layer_advantage_by_circuit_depth.csv", stringsAsFactors = FALSE)

# data <- data |> filter(algorithm != "Random")
# data$algorithm[data$algorithm == "Random"] <- "Random guessing"
data <- data %>% mutate(algorithm = fct_relevel(algorithm, 
    "QAOA",
    "WSQAOA",
    "WS-Init-QAOA",
    "RQAOA",
    # "Random guessing"
))

data$n_qaoa_layers <- as.factor(data$n_qaoa_layers)

# data <- data |> filter(algorithm != "RQAOA")

# data <- data |>
#     group_by(algorithm, n_qaoa_layers, problem) |>
#     filter(
#         between(performance, quantile(perfocmance, 0.25), quantile(performance, 0.75))
#     ) |>
#     ungroup()

combine <- TRUE


if (combine) {
    data <- data |>
        group_by(algorithm, n_qaoa_layers, problem, n_qubits) |>
        summarize(runtime = mean(runtime), performance = mean(performance), .groups = "drop")
}

if (combine) {
    alpha <- 1
    size <- 0.5
} else {
    alpha <- 0.2
    size <- 0.1
}

g <- ggplot(data=data, aes(x=runtime, y=performance, color=n_qaoa_layers, shape=algorithm, size=n_qubits)) +
    # geom_smooth(linewidth=0.1, se=FALSE, method="lm", aes(group_by=n_qubits)) +
    geom_point(alpha=alpha) +
    scale_color_manual(name="\\# Layers", values=COLOURS.LIST) +
    scale_size(name=NULL, range=c(0.2, 1.0)) +
    scale_shape_manual(values=c(1, 3, 4, 2)) +
    scale_x_log10(
        breaks=c(0.25, 0.5, 1, 2, 4, 8, 16),
        minor_breaks=c()
    ) +
    # scale_x_sqrt(
    #     breaks=c(1, 2, 5, 10, 20),
    #     minor_breaks=c()
    # ) +
    # scale_x_sqrt() +
    scale_y_continuous() +
    # scale_color_manual(values=light_colors) +
    # guides(colour=guide_legend(override.aes=list(alpha=1, size=1, shape=4))) +

    # geom_point(data=average_data, size=0.5, alpha=1, shape=4, aes(color=problem)) +
    facet_grid2(~ problem) +
    # theme(plot.margin=unit(c(0,0,0,0), 'cm')) +
    # theme(legend.margin=unit(-1, 'cm')) +
    # scale_linetype_manual(values=c("solid", "11")) +
    # scale_linetype(name="Algorithm") +
    # scale_x_continuous(minor_breaks=NULL) +
    # scale_y_continuous(breaks=seq(0.8, 1.2, 0.1)) +
    # guides(color=guide_legend()) +
    guides(
        size="none",
        color=guide_legend(order=1),
        shape=guide_legend(order=2, title=NULL)
    ) +
    labs(y="Approximation Quality", x="Runtime [s, log]") +
    theme_paper_base() +
    theme(
        legend.spacing = unit(0, "cm"),
        legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm")
    ) + 
    theme(
        legend.box = "vertical",
        legend.title = element_text(size=BASE.SIZE),
        # strip.text.y = element_text(size=4),
        # strip.text.x = element_text(size=6),
    )

    # guides(color=guide_legend(title.position="top")) +

save_name <- "performance_by_runtime"
height_factor = 0.6
width_factor = 2
pdf(str_c(PDF_OUTDIR, save_name, ".pdf"), width = width_factor * COLWIDTH, height = height_factor*COLWIDTH)
print(g)
dev.off()
tikz(str_c(TIKZ_OUTDIR, save_name, ".tex"), width = width_factor * COLWIDTH, height = height_factor*COLWIDTH)
print(g)
dev.off()
