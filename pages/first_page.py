import os
from PIL import Image
import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
import base64
from PIL import Image
import io

auth = ClarifaiAuthHelper.from_streamlit(st)
print(auth._pat)
stub = auth.get_stub()
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')
print(userDataObject)

def model_predict_by_file(image_bytes,
                          MODEL_ID):
    
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataClarifaiMain,
            # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=image_bytes
                        )
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_model_outputs_response

def model_text_predict(auth_obj,
                       MODEL_ID,
                       RAW_TEXT):
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=RAW_TEXT
                        )
                    )
                )
            ]
        ),
        metadata=auth_obj
    )
    return post_model_outputs_response

def workflow_predict(WORKFLOW_ID,
                     INPUT_TEXT):  # The userDataObject is required when using a PAT

    post_workflow_results_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=userDataObject,
            workflow_id=WORKFLOW_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=INPUT_TEXT
                        )
                    )
                )
            ]
        ),
        metadata = (('authorization', 'Key ' + auth._pat),)

    )
    return post_workflow_results_response


if __name__ == '__main__':    

    # clarifai_logo = "pages/Clarifai_Logo_FC_Logo.jpg"
    # image = Image.open(clarifai_logo)
    # st.image(image, use_column_width=True, caption="Clarifai Logo", width=100)

    st.title("Generative AI and Clarifai - Stable Diffusion")
    with st.sidebar:
        # st.image(image, use_column_width=True, caption="Clarifai Logo", width=100)
        text = st.text_input("Input the text to generate the image from")
        model_options = st.multiselect("Select the models to run the generated image through",
                                       ["general-english-image-caption-clip",
                                        "general-image-recognition"])
        inp_button = st.button("**Generate Image and run through the models selected**")

        if inp_button:
            st.write(f"**Text submitted for generating image:** \n{text}")
            # st.write(text)
            st.write(f"**Models Selected:**")
            for i in range(len(model_options)):
                st.write(f"{i+1}. {model_options[i]}")

    if inp_button:
        with st.spinner("Generating Image..."):
            workflow_payload = workflow_predict( WORKFLOW_ID="stable-diffusion-xl",
                                                 INPUT_TEXT=text)
            image_b64 = workflow_payload.results[0].outputs[0].data.image.base64   
            image = Image.open(io.BytesIO(image_b64))
            # image.save("generated_image.jpg")
            st.image(image, use_column_width=True, caption="Generated Image", width=100)
            st.success("Image Generated Successfully")

        # With image saved now run it through models in prod

        for i in range(len(model_options)):
            with st.spinner(f"predicting to model {model_options[i]}"):
                if model_options[i] == "general-english-image-caption-clip":
                    model_id = "general-english-image-caption-clip"
                    model_payload = model_predict_by_file(image_bytes=image_b64, MODEL_ID=model_id)
                    st.write(f"**Model: {model_options[i]}**")
                    st.write(f"**Image Caption:**")
                    st.write(model_payload.outputs[0].data.text.raw)
                elif model_options[i] == "general-image-recognition":
                    df = pd.DataFrame(columns=["Concept", "Probability"])
                    model_id = "general-image-recognition"
                    model_payload = model_predict_by_file(image_bytes=image_b64, MODEL_ID=model_id)
                    st.write(f"**Model: {model_options[i]}**")
                    st.write(f"**Image Tags:**")

                    for j in range(len(model_payload.outputs[0].data.concepts)):
                        # df = df.append({"Concept": model_payload.outputs[0].data.concepts[j].name,
                        #                 "Probability": model_payload.outputs[0].data.concepts[j].value},
                        #                ignore_index=True)
                        print(model_payload.outputs[0].data.concepts[j].name,':', model_payload.outputs[0].data.concepts[j].value)
                    # st.write(df)
                st.write("----"*60)
                        # st.write(f"{j+1}. {model_payload.outputs[0].data.concepts[j].name}")

