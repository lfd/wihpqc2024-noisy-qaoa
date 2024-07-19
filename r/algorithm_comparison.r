library(tidyverse)
library(ggplot2)
library(tikzDevice)
library(ggh4x)

source("r/base.r")

options(
    tikzLatexPackages = c(
        getOption("tikzLatexPackages"),
        "\\usepackage{amsmath}" # , "\\usepackage[utf8]{inputenc}"
    )
)

do.save.tikz <- function(g, out.name, .width, .height) {
    tikz(str_c("img-tikz/", out.name, ".tex"), width = .width, height = .height)
    print(g)
    dev.off()
}
do.save.pdf <- function(g, out.name, .width, .height) {
    pdf(str_c("img-pdf/", out.name, ".pdf"), width = .width, height = .height)
    print(g)
    dev.off()
}


COLOURS.LIST <- c("black", "#E69F00", "#999999", "#009371",
                  "#beaed4", "#ed665a", "#1f78b4", "#009371")
FILLS.LIST = c("white", "green")


dat_n_layers <- read_csv("csvs/algorithm_comparison_n_layers.csv")
dat_n_qubits <- read_csv("csvs/algorithm_comparison_n_qubits.csv")

dat_n_layers$n <- dat_n_layers$n_qaoa_layers
dat_n_qubits$n <- dat_n_qubits$n_qubits

dat_n_layers$n_qaoa_layers <- NULL
dat_n_qubits$n_qubits <- NULL

dat_n_layers$type <- "By layers"
dat_n_qubits$type <- "By qubits"



dat <- rbind(dat_n_layers, dat_n_qubits)

dat.sub <- dat %>% filter(!is.na(algorithm))
dat.base <- dat %>% filter(is.na(algorithm)) %>% select(-algorithm) %>% rename(bound=model)

rects <- read_csv("csvs/quantum_advantages.csv")
rects$model <- "Ideal"
rects$performance <- 1
rects$n <- 5
rects$n[rects$type == "By layers"] <- 1
rects$advantage <- factor(rects$advantage)

g <- ggplot(dat.sub, aes(x=n, y=performance,
                colour=model, shape=model)) +
    scale_colour_manual("Simulation", values=COLOURS.LIST) +
    scale_shape_discrete("Simulation") +
    scale_x_continuous(breaks=c(1, 2, 3, 4, 6, 8, 10)) +
    geom_rect(
        data = rects,
        aes(fill = advantage),
        alpha = 0.07,
        xmin = -Inf,xmax = Inf, ymin = -Inf,ymax = Inf,
        show.legend = FALSE
    ) +
    scale_fill_manual(values=FILLS.LIST) +
    geom_line(dat=dat.base, inherit.aes=FALSE, 
              aes(x=n, y=performance, group=bound, linetype=bound),
              colour=COLOURS.LIST[[3]]) +
    geom_line() + geom_point(size=0.8) + theme_paper_base() +
    facet_nested(algorithm ~ problem + type, scales="free_x") +
    xlab("\\# QAOA layers $p$ resp. \\# Qubits $n$") + ylab("Approximation Quality") +
    scale_linetype_discrete("Bound") + theme(
        legend.box="vertical",
        legend.spacing=unit(c(0), "cm"),
        legend.title=element_text(size=BASE.SIZE),
        strip.text.y = element_text(size=4.5),
        strip.text.x = element_text(size=6.5),
    )

WIDTH <- COLWIDTH
HEIGHT <- WIDTH * 0.9
do.save.pdf(g, "algorithm_comparison", WIDTH, HEIGHT)
do.save.tikz(g, "algorithm_comparison", WIDTH, HEIGHT)
