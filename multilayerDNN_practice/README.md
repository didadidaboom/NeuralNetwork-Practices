# Multi-layer DNN practice

+ Workspace architecture

  ~~~shell
  .					# root
  ├── config.py			# configure file (generally, yaml)
  ├── data			# dataset file
  ├── dataset_banknote.py		# script of dataset dataloader
  ├── inference.py		# inferece script
  ├── log				# log directory
  ├── loss.py			# loss design script
  ├── model.py			# model design scrip
  ├── trainer.py			# trainer scrip
  ├── model_save			# model (checkpoint) directory
  ├── preprocess.py		# for dataset processing
  └── README.md			# readme
  ~~~

+ Topic: Identification of real and fake banknote based on multi-layer DNN model

+ Goals:

  - Familiar with the flowchart of how to build a model by pytorch

  - Identyfying real and fake banknote by multi layer DNN model

