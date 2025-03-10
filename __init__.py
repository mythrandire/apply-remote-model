import os
from datetime import datetime
import torch
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos


TRAIN_ROOT = '/tmp/yolo_train'
MODEL_ROOT = os.path.join(TRAIN_ROOT, 'models')
DATA_ROOT = os.path.join(TRAIN_ROOT, 'data')
PROJECT_ROOT = os.path.join(TRAIN_ROOT, 'projects')



#
# We assume you have ultralytics installed
#
#try:
#    from ultralytics import YOLO
#except ImportError:
#    raise ImportError(
#        "You must install ultralytics to use this plugin. "
#        "Add `ultralytics` to your plugin's requirements.txt."
#    )


class ApplyRemoteModel(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name="apply-remote-model",  # Must match what's in fiftyone.yml
            label="Run YOLO model with cloud-backed weights",
            description="Run inference with a YOLOv8 model on the current view using remotely stored weights",
            icon="input",           # Material UI icon, or path to custom icon
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=False,
        )

    def resolve_placement(self, ctx):
        """
        Optional convenience: place a button in the App so the user can
        click to open this operator's input form.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Apply YOLO model",
                icon="input",
                prompt=True,  # always show the operator's input prompt
            ),
        )

    def resolve_input(self, ctx):
        """
        Collect the inputs we need from the user. This defines the form
        that appears in the FiftyOne App when the operator is invoked.
        """
        inputs = types.Object()


        dataset = ctx.dataset
        schema = dataset.get_field_schema(ftype=fo.EmbeddedDocumentField,
                                          embedded_doc_type=fo.Detections)
        fields = schema.keys()
        field_choices = types.DropdownView()
        for field_name in fields:
            field_choices.add_choice(field_name, label=field_name)

        inputs.str(
            'det_field',
            required=True,
            label='Detections field',
            view=field_choices,
        )

        # 1) Local filepath to existing YOLOv8 model weights
        inputs.str(
            "weights_path",
            default='gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt',
            required=True,
            description="Filepath to the YOLOv8 *.pt weights file",
            label="YOLOv8 weights",
        )

        # 2) CUDA target device
        inputs.int(
            "target_device_index",
            default=0,
            required=False,
            description='CUDA Device number to train on. Optional, defaults to device cuda:0',
            label="Target CUDA device number"
        )

        return types.Property(
            inputs,
            view=types.View(label="Run inference YOLOv8"),
        )

    def execute(self, ctx):
        """
        """
        
        from ultralytics import YOLO

        det_field = ctx.params["det_field"]
        weights_path = ctx.params["weights_path"]
        target_device_index = ctx.params["target_device_index"]

        dataset = ctx.dataset

        # --- Step 1: Verify the weights_path is YOLOv8 ---
        local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
        fos.copy_file(weights_path, local_weights_path)
        #model = self._try_load_model(local_weights_path)
        str = f'Model downloaded to: {local_weights_path}'
        ctx.log(str)
        print(str)

        cuda_device_count = torch.cuda.device_count()
        ctx.log(f"Number of CUDA devices found: {cuda_device_count}")

        model = YOLO(local_weights_path)
        
        if cuda_device_count > 1 and target_device_index <= cuda_device_count:
            target_device = f"cuda:{target_device_index}"
            model.to(target_device)
        else:
            model.to("cuda:0")

        ctx.dataset.apply_model(model, label_field=det_field)

        print("Ending inference")

        #ctx.set_progress(progress=1.0, label="Done!")
        return {
            "status": "success",
            "cuda_device_count": cuda_device_count
        }

    def resolve_output(self, ctx):
        """
        Display any final outputs in the App after training completes.
        """
        outputs = types.Object()
        outputs.str(
            "status",
            label="Finetuning status",
        )

        outputs.str(
            "cuda_device_count",
            label="Number of CUDA devices"
        )
        return types.Property(
            outputs,
            view=types.View(label="Finetune Results"),
        )

def register(plugin):
    """
    Called by FiftyOne to discover and register your plugin’s operators/panels.
    """
    plugin.register(ApplyRemoteModel)
    