BASE.SIZE <- 9
COLOURS.LIST <- c("black", "#E69F00", "#999999", "#009371", "#beaed4", "#ed665a", "#1f78b4", "#009371")
PT.PER.INCH = 1/72.27
COLWIDTH=252.0 * PT.PER.INCH

theme_paper_base <- function() {
    return(theme_bw(base_size=BASE.SIZE) +
           theme(axis.title.x = element_text(size = BASE.SIZE),
                 axis.title.y = element_text(size = BASE.SIZE),
                 legend.title = element_blank(),
                 legend.position = "top",
                 plot.margin = unit(c(0,0,0,0), "cm"),
                 legend.margin = margin(0,0,0,0),
                legend.box.margin=unit(c(0,0,0,0), "cm")
            )
        )
                                  
}

TIKZ_OUTDIR <- "img-tikz/"
PDF_OUTDIR <- "img-pdf/"