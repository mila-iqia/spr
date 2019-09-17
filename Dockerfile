FROM philly/jobs/custom/generic-docker:py36
RUN conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch
RUN pip install matplotlib wandb wget opencv-python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN wandb login fa235c4236ea0bd894bc3f26e87054a6ed6af293