FROM jupyter/datascience-notebook:latest

USER root

# install required libraries

RUN apt-get update \
  && apt-get install -y \
      vim curl wget git zip unzip bash iputils-ping net-tools \
      fonts-ipaexfont

RUN pip install --no-cache-dir \
  pulp \
  backports.tarfile

# matplotlibで日本語を使えるようにする
RUN mpl_path=$(python -c "import matplotlib; print(matplotlib.matplotlib_fname())") && \
    sed -i '/font\.family/d' "$mpl_path" && \
    echo "font.family: IPAexGothic" >> "$mpl_path" && \
    rm -rf /home/jovyan/.cache

USER jovyan
