# -*- mode: ruby -*-
# vi: set ft=ruby :
Vagrant.configure(2) do |config|

  config.vm.define "nlp_projcet" do |nlp_projcet|
      nlp_projcet.vm.box = "ubuntu/xenial64"
      nlp_projcet.vm.network "private_network", ip: "192.168.33.10"
      nlp_projcet.vm.synced_folder "./", "/vagrant", owner: "ubuntu", mount_options: ["dmode=775,fmode=775"]
      nlp_projcet.vm.provider "virtualbox" do |vb|
        # Customize the amount of memory on the VM:
        vb.memory = "2048"
        vb.cpus = 2
      end
  end

  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y git build-essential
    # using the python3
    sudo apt-get install -y python3 python3-pip python3-dev
    pip3 install --upgrade pip3
    apt-get -y autoremove
    # Install NLTK and ML tools dependencies
    cd /vagrant
    sudo pip3 install -r requirements.txt
    sudo pip3 install ipython
    # install pyenv
    sudo curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
    # install pytorch
    pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
    pip3 install torchvision
    pip3 install torchtext
    sudo apt-get install -y python python-pip python-dev
    sudo pip install -r requirements.txt
    git clone https://github.com/harvardnlp/boxscore-data
    git clone https://github.com/harvardnlp/data2text
  SHELL
end
