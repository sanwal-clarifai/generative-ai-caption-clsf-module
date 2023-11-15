import os
from PIL import Image
import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
import base64
from PIL import Image
import io

channel = ClarifaiChannel.get_json_channel(base_url='https://api-dev.clarifai.com/')
stub = service_pb2_grpc.V2Stub(channel)

channel_prod = ClarifaiChannel.get_grpc_channel()
stub_prod = service_pb2_grpc.V2Stub(channel_prod)

def model_predict_by_file(auth_obj,
                          image_path,
                          MODEL_ID):
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    post_model_outputs_response = stub_prod.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObjectProd,
            # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=auth_obj
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

def workflow_predict(auth_obj,
                     WORKFLOW_ID,
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
        metadata=auth_obj
    )
    return post_workflow_results_response


if __name__ == '__main__':
    pat = "54febaf57e214497ad5bb380b45abc73"  # DEV PAT
    pat_prod = "a55a5b7c56df4ef6a111d7fbd5f5d059"  # PROD PAT

    api_auth = (('authorization', 'Key ' + pat),)
    api_auth_prod = (('authorization', 'Key ' + pat_prod),)

    APP_ID = "sd-gai-text"
    USER_ID = "sanwalyousaf-dev"
    # APP_ID = "stable-diffusion"
    # USER_ID = "pv9"

    APP_ID_PROD = "pepsi-demo-app"
    USER_ID_PROD = "sanwalyousaf"

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID,
                                                app_id=APP_ID)
    userDataObjectProd = resources_pb2.UserAppIDSet(user_id=USER_ID_PROD,
                                                    app_id=APP_ID_PROD)

    ####Make predictions to the Stable  Diffusion Workflow in Dev

    workflow_id = "sd-text-clsf"

    clarifai_logo = "Clarifai_Logo_FC_Logo.jpg"
    image = Image.open(clarifai_logo)
    # st.image(image, use_column_width=True, caption="Clarifai Logo", width=100)

    st.title("Generative AI and Clarifai - Stable Diffusion")


    with st.sidebar:
        st.image(image, use_column_width=True, caption="Clarifai Logo", width=100)
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
            workflow_payload = workflow_predict(api_auth, workflow_id, text)

            image_b64 = workflow_payload.results[0].outputs[0].data.image.base64
            # b64_decode = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_b64))
            image.save("generated_image.jpg")
            st.image(image, use_column_width=True, caption="Generated Image", width=100)
            st.success("Image Generated Successfully")

        # With image saved now run it through models in prod
        for i in range(len(model_options)):
            with st.spinner(f"predicting to model {model_options[i]}"):
                if model_options[i] == "general-english-image-caption-clip":
                    model_id = "general-english-image-caption-clip"
                    model_payload = model_predict_by_file(api_auth_prod, "generated_image.jpg", model_id)
                    st.write(f"**Model: {model_options[i]}**")
                    st.write(f"**Image Caption:**")
                    st.write(model_payload.outputs[0].data.text.raw)
                elif model_options[i] == "general-image-recognition":
                    df = pd.DataFrame(columns=["Concept", "Probability"])
                    model_id = "general-image-recognition"
                    model_payload = model_predict_by_file(api_auth_prod, "generated_image.jpg", model_id)
                    st.write(f"**Model: {model_options[i]}**")
                    st.write(f"**Image Tags:**")

                    for j in range(len(model_payload.outputs[0].data.concepts)):
                        df = df.append({"Concept": model_payload.outputs[0].data.concepts[j].name,
                                        "Probability": model_payload.outputs[0].data.concepts[j].value},
                                       ignore_index=True)
                    st.write(df)
                st.write("----"*60)
                        # st.write(f"{j+1}. {model_payload.outputs[0].data.concepts[j].name}")

