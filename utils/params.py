import os
import glob
from dataset.model_net_40 import ModelNet40

class Params:
    def __init__(self, **kwargs):
        ## Device
        self.device='cuda'

        self.model='DGCNN'
        self.epochs=120
        self.num_points=1024
        self.emb_dims=1024 #Dimension of embeddings
        self.k=20 #Num of nearest neighbors to use
        self.optimizer='ADAM'
        self.lr=0.0001# learning rate (default: Adam=0.001, SGD=0.1)
        self.momentum=0.9
        self.dropout=0.5
        self.att_heads=8

        ## Logging and history
        self.save_checkpoint=True
        self.dataset='ModelNet40'
        self.dump_file=True
        self.dry_ryn=False
        self.eval=False
        self.dataset_loader=ModelNet40

        # DataSet
        self.number_classes=40
        self.batch_size=32
        self.test_batch_size=16
        self.num_workers=8
        self.random_state=42

        for attr_name in kwargs.keys():
            setattr(self,attr_name,kwargs[attr_name])

        self.output_dir='./tmp/output/{}'.format(self.model)
        self.execution_id=self.generate_execution_id(self.output_dir, self.dataset)

        self.number_classes=40
        # if self.dataset_loader == ModelNet40:
        #     self.number_classes=40
        # else:
        #   self.number_classes=10
        self.setup_log_structure()

    def generate_execution_id(self, path, dataset):
        executions = [f for f in glob.glob(f'{path}/{dataset}/execution_*.log')]
        return 'execution_{}'.format('{:04d}'.format(len(executions) + 1))

    def setup_log_structure(self):
        if not os.path.exists('{}/{}'.format(self.output_dir, self.dataset)):
            os.makedirs('{}/{}'.format(self.output_dir, self.dataset))

        if not os.path.exists('{}/{}/checkpoints'.format(self.output_dir, self.dataset)):
            os.makedirs('{}/{}/checkpoints'.format(self.output_dir, self.dataset))

        if not os.path.exists('{}/{}/checkpoints/{}'.format(self.output_dir, self.dataset,self.execution_id)):
            os.makedirs('{}/{}/checkpoints/{}'.format(self.output_dir, self.dataset,self.execution_id))

    def last_checkpoint(self):
        checkpoints = self.list_checkpoints()
        return '' if len(checkpoints) == 0 else checkpoints[-1]

    def list_checkpoints(self):
        return [f for f in glob.glob('{}/{}/checkpoints/{}/*.t7'.format(self.output_dir, self.dataset, self.execution_id))]

    def checkpoints_count(self):
        return len(self.list_checkpoints())

    def checkpoint_path(self):
        return '{}/{}/checkpoints/{}/model_{}.t7'.format(self.output_dir, self.dataset, self.execution_id, '{:04d}'.format(self.checkpoints_count() + 1))

    def log(self, content, p=True):
        if p:
            print(content)
        if self.dump_file:
            with open("{}/{}/{}.log".format(self.output_dir, self.dataset, self.execution_id), "a") as f:
                f.write(content)
                f.write('\n')

    def csv_path(self):
        return "{}/{}/{}.csv".format(self.output_dir, self.dataset, self.execution_id)

    def plot_path(self):
        return "{}/{}/{}.png".format(self.output_dir, self.dataset, self.execution_id)

    def csv(self, epoch, train_loss, train_acc, train_avg_acc, validation_loss, validation_acc, validation_avg_acc, time):
        self.log('Train: %d, time: %.6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, time, train_loss,train_acc,train_avg_acc))
        self.log('Validation: %d, time: %.6f, loss: %.6f, validation acc: %.6f, validation avg acc: %.6f' % (epoch, time, validation_loss, validation_acc, validation_avg_acc))

        if self.dump_file:
            print_header = not os.path.isfile(self.csv_path())

            with open(self.csv_path(), "a") as f:
                if print_header:
                    f.write("epoch,train_loss,train_acc,train_avg_acc,validation_loss,validation_acc,validation_avg_acc,time\n")
                f.write("{},{},{},{},{},{},{}".format(epoch, train_loss, train_acc, train_avg_acc, validation_loss, validation_acc, validation_avg_acc, time))
                f.write('\n')

    def print_summary(self, validation_loss, validation_acc, validation_avg_acc):
        if self.dump_file:
            print_header = not os.path.isfile("{}/{}/summary.txt".format(self.output_dir, self.dataset))
            with open("{}/{}/summary.txt".format(self.output_dir, self.dataset), "a") as f:
                if print_header:
                    f.write("execution_id,model,dataset,batch_size,test_batch_size,epochs,att_heads,optimizer,learning_rate,momentum,num_points,dropout,emb_dims,k,loss,validation_acc,validation_avg_acc\n")
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                    self.execution_id,
                    self.model,
                    self.dataset,
                    self.batch_size,
                    self.test_batch_size,
                    self.epochs,
                    self.att_heads,
                    self.optimizer,
                    self.lr,
                    self.momentum,
                    self.num_points,
                    self.dropout,
                    self.emb_dims,
                    self.k))
                f.write(",{},{},{}\n".format(
                    validation_loss,
                    validation_acc,
                    validation_avg_acc))
