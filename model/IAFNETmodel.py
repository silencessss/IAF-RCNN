import os
import Base_model


C = config.Config()

class Base_model():
    def name(self):
        return 'BaseModel'
    def initialize(self,opt):
        Base_model.initialize(self,opt)
		# specify the training details
		self.cls_loss_r1 = []
		self.regr_loss_r1 = []
		self.cls_loss_r2 = []
		self.regr_loss_r2 = []
		self.illuminate_loss = []
		self.losses = np.zeros((self.epoch_length, 9))
		self.optimizer = Adam(lr=opt.init_lr)
		print ('Initializing the {}'.format(self.name()))

    def create_IAFNet_model(self,opt,train_data,phase='train'):
        '''
        :|opt:config
        :|train_data: list
        '''
        Base_model.crate_base_IAFNet_model()
        illuminate_value = self.illuminate_output
        illuminate_value_teacher = self.illuminate_output_teacher
        sefl.model_teacher = Model([self.img_input_rgb,self.img_input_lwir],illuminate_value_teacher)
 
        if phase=='train':
            self.model = Model([self.img_input_rgb,self.img_input_lwir],illuminate_value)
            self.model.compile(optimizer=self.optimizer,loss=[losses.illumination_loss],sample_weight_mode=None)
        self.model_all = Model([self.img_input_rgb,self.img_input_lwir],illuminate_value)

    def train_IAFNet(self,opt,out_path):
        iter_num=0
        start_time=time.time()
        
        for epoch_num in range(self.num_epochs):
            progbar = generic_utils.Progbar(self.epoch_length)
            print('Epoch {}/{}'.format(epoch_num + 1 + self.add_epoch, self.num_epochs + self.add_epoch))
            :???:K.get_value
            '''
            :||learning rate change||:
            '''
            if  epoch_num % C.reduce_lr_epoch == 0 and epoch_num != 0 :
                lr = K.get_value(sefl.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr,0.1*lr)
                lr_later = K.get_value(self.model.optimizer.lr)
                print("model lr changed to {}".format(lr_later))
            while True:
                try:
                    :||:data_gen_train from base_model.py
                    # :|| Y is result ||:
                    [x,x_lwir],Y,img_data,illumination_batch_value = next(self.data_gen_train)
                    # :|| illlumination ||:
                    loss_s0 = self.model.train_on_batch([X,X_lwir],illumination_batch_value)
                    sefl.losses[iter_num,4] = loss_s0
                    # :|| cls1 and regr1 ||:

                    # :|| cls2 and regr2 ||:



                    :||:no apply the weight:||:
                    
                    iter_num+=1

                    # :|| update the training information ||:
                    if iter_num%20 == 0:
                        progbar.update(
                            iter_num,
                            [
                                ('cls1',np.mean(self.losses[:iter_num,0])),
                                ('regr1',np.mean(self.losses[:???:])),
                                ('cls2',np.mean(self.losses[:???:])),
                                ('regr2',np.mean(self.losses[:???:])),
                                ('illuminate',np.mean(self.losses[:iter_num,4])),
                                ('lr',lr_later)
                            ]
                        )
                    # :|| save the trianing weights ||:
                    if iter_num == (self.epoch_length//4) or iter_num == (sefl.epoch_length//2) or iter_num == (self.epoch_length //4)*3 or iter_num == (self.epoch_length):
                        self.model_teacher.save_weights(os.path.join(out_path,'resnet_e{}_l{}.hdf5'.format(epoch_num + 1 + self.add_epoch, iter_num)))

                    # :|| update and recoder the training information ||:
                    if iter_num == self.epoch_length:
                        cls_loss1 = np.mean(self.losses[:, 0])
						regr_loss1 = np.mean(self.losses[:, 1])
						cls_loss2 = np.mean(self.losses[:, 2])
						regr_loss2 = np.mean(self.losses[:, 3])
						illuminate_loss = np.mean(self.losses[:, 4])
						total_loss = cls_loss1 + regr_loss1 + cls_loss2 + regr_loss2 + illuminate_loss

                        self.total_loss_r.append(total_loss)
                        self.cls_loss_r1.append(cls_loss1)
                        self.regr_loss_r1.append(regr_loss1)
                        self.cls_loss_r2.append(cls_loss2)
						self.regr_loss_r2.append(regr_loss2)
						self.illuminate_loss.append(np.mean(self.losses[:, 4]))
                        print('Total loss: {}'.format(total_loss))
						print('Elapsed time: {}'.format(time.time() - start_time))

                        iter_num = 0
						start_time = time.time()

						if total_loss < self.best_loss:
							print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, total_loss))
							self.best_loss = total_loss

						break
                    except Exception as e:
                        print ('Exception: {}'.format(e))
					    continue
                records = np.concatenate((np.asarray(self.total_loss_r).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r1).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r1).reshape((-1, 1)),
									  np.asarray(self.cls_loss_r2).reshape((-1, 1)),
									  np.asarray(self.regr_loss_r2).reshape((-1, 1)),
									  np.asarray(self.illuminate_loss).reshape((-1, 1))),
									 axis=-1)
			    np.savetxt(os.path.join(out_path, 'records.txt'), np.array(records), fmt='%.8f')
		print('Training complete, exiting.')

                







