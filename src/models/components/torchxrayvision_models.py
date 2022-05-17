from torch import nn
import torchvision
import pytorch_lightning as pl
import torchxrayvision as xrv


class XRVModels(pl.LightningModule):
    """
    Models are defined in the following way    

    num_classes is no longer a custom input (https://github.com/mlmed/torchxrayvision/issues/59)
    num_classes = 18 (set by default)
    """

    def __init__(self, hparams: dict):
        XRV_NUM_CLASSES=18
        super().__init__()

        if hparams["model"].lower()=="rsna":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-rsna",
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="nih":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-nih", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="padchest":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-pc", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="chexpert":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-chex", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="mimic_nb":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="mimic_ch":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="all":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all", 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet121":
            self.model = torchvision.models.densenet121(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet169":
            self.model = torchvision.models.densenet169(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet161":
            self.model = torchvision.models.densenet161(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet201":
            self.model = torchvision.models.densenet201(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        # Modifying classifiers to current number of classes
        if hparams["model"].lower() in ["densenet121","densenet169","densenet161","densenet201", "googlenet"]:
            features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(features, 500),
                nn.Linear(500, XRV_NUM_CLASSES))

    def forward(self, x):
        return self.model(x)