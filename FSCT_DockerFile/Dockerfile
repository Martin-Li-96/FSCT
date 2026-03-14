FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-devel-ubi9 
COPY rocky.repo /etc/yum.repos.d
COPY requirements.txt /tools/requirements.txt
COPY FSCT /FSCT
ENV PIP_NO_CACHE_DIR=1
RUN rm -rf /etc/yum.repos.d/redhat.repo &&\
    rm -rf /etc/yum.repos.d/ubi.repo &&\
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm &&\
    rpm -e --nodeps openssl-fips-provider &&\
    rpm -e --nodeps openssl-fips-provider-so &&\
    dnf clean all &&\
    dnf distro-sync -y &&\
    dnf update -y &&\
    dnf install -y ncurses procps openssh-server sudo mesa-libGL \
                   mesa-dri-drivers python39 python3-devel freetype freetype-devel python3-tkinter &&\
    dnf clean all 


RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129 

RUN pip install torch_geometric &&\
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html &&\
    pip install -r /tools/requirements.txt 

RUN echo "/usr/local/lib/python3.9/site-packages/nvidia/cusparselt/lib" > /etc/ld.so.conf.d/cusparselt.conf &&\
    ldconfig &&\
    rm -rf /root/.cache &&\
    rm -rf /tmp/*



