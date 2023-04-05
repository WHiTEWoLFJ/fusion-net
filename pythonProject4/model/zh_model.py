
from torch.autograd import Variable
from .base_model import BaseModel
from .network import CFMSAN




class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C_gray = self.Tensor(nb, opt.output_nc, 600, 400)

        skip = True if opt.skip > 0 else False

        self.d_net = CFMSAN()
        self.d_net.cuda()
        self.h_net = CFMSAN()
        self.h_net.cuda()
        if not self.isTrain or opt.continue_train:
            print("---is not train----")
            which_epoch = opt.which_epoch
            print("---model is loaded---")
            #self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.d_net, 'G_V', which_epoch)
            self.load_network(self.h_net, 'G_H', which_epoch)
        print('---------- Networks initialized -------------')
        self.d_net.eval()
        self.h_net.eval()
        print('-----------------------------------------------')
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        if self.opt.isTrain:
            input_C = input['C']
            input_C_gray = input['C_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        if self.opt.isTrain:
            self.input_C.resize_(input_C.size()).copy_(input_C)
            self.input_C_gray.resize_(input_C_gray.size()).copy_(input_C_gray)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_image_paths(self):
        return self.image_paths
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        self.real_C_gray = Variable(self.input_C_gray)
        self.a1=self.d_net.forward(self.real_A)
        self.a2=self.h_net.forward(self.real_B)
        self.output1=self.a1*self.real_A+self.real_B*self.a2




