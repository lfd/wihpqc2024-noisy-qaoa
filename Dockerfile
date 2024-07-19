FROM rocker/tidyverse

RUN R -e "install.packages(c('tikzDevice', 'cowplot', 'gridExtra', 'ggh4x', 'ggnewscale'))"
RUN apt-get update && apt-get install -y texlive-latex-extra

WORKDIR /app
COPY r /app/r
COPY csvs /app/csvs

CMD Rscript r/all_scripts.r