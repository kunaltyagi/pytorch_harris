.PHONY: clean all source abs no_abs

all: out/model.blob

clean:
	rm out -fr

SHELL:=bash
OPENVINO_DIR=${HOME}/.local/opt/intel/openvino
TOOLS_DIR=${OPENVINO_DIR}/deployment_tools

out/model.onnx: model.py
	source ~/virtualenv/ml/bin/activate && \
	python model.py

no_abs:
	source ~/virtualenv/ml/bin/activate && \
	python model.py

abs:
	source ~/virtualenv/ml/bin/activate && \
	python model.py --abs

out/model.xml: out/model.onnx
	source ~/virtualenv/ml/bin/activate && \
	python3 ${TOOLS_DIR}/model_optimizer/mo_onnx.py --input_model "out/model.onnx" --data_type half -o out --input_shape "[1, 3, 300, 300]"

out/model.blob: out/model.xml
	. ${OPENVINO_DIR}/bin/setupvars.sh && \
	${TOOLS_DIR}/tools/compile_tool/compile_tool -m out/model.xml -o out/model.blob -ip U8 -d MYRIAD -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4
