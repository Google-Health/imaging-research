# Derm Foundation

**Derm Foundation** is a tool to generate
[embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
from dermatological images. These embeddings can be used to develop custom
machine learning models for dermatology use-cases with less data and compute
compared to traditional model development methods.

## How to use the Derm Foundation API

1.  Decide if you want to get access as an individual or group. For more information see [Access Options](#access-options)

1.  With the individual or group email identity at hand from the previous step,
    fill out the [API access form](https://forms.gle/VBFuzSJXhQjNmF776).

1.  Once access is granted, you’ll be notified via the provided email address
    and can start using the API.

1.  The [Demo Notebook](https://colab.research.google.com/github/Google-Health/imaging-research/blob/master/derm-foundation/derm_foundation_demo.ipynb) shows you how to use the API to
    train a sample model [with our test data](#use-our-test-data). You can
    modify the Demo Notebook to train a model using
    [your own data](#use-your-own-data). This Notebook provides an example of
    the following steps:

    *   Generating a temporary access token to grant the API access to images in
        GCS.
    *   Calling the API with a given GCS bucket name, GCS object path, and the
        access token.
    *   Saving the embedding.
    *   Using the embeddings to train a simple model.
    *   Evaluate the results of the model

1.  If you need support or have questions, please [contact us](#contact-us).

## Use our test data

Upon gaining access to the API, you'll also have access to publicly available
data we've curated specifically for testing. This is to help you get started
with your initial experiments. The default state of the
[Demo Notebook](https://colab.research.google.com/github/Google-Health/imaging-research/blob/master/derm-foundation/derm_foundation_demo.ipynb) is set to use this test data, which is
stored in a
[Cloud Storage (GCS) bucket](https://cloud.google.com/storage/docs/creating-buckets)
managed by us for your convenience.

## Use your own data

WARNING: You hold responsibility for the data stored in your GCS bucket that you
use with the API. It's important to comply with all the terms of use any data is subject to.

NOTE: The [Demo Notebook](https://colab.research.google.com/github/Google-Health/imaging-research/blob/master/derm-foundation/derm_foundation_demo.ipynb) demonstrates how to call the
API using short-lived access tokens. These tokens provide temporary access to
the API for processing your images and are specific to the individual running
the Colab. It's important to note that the API is stateless and does not store
the images it processes.

1.  If you don't have access to an existing
    [GCP Project](https://cloud.google.com/storage/docs/projects), you need to
    [create one](https://cloud.google.com/free).

1.  [Create a GCS bucket](https://cloud.google.com/storage/docs/creating-buckets)
    in the above project.

1.  On your local machine
    [install the gcloud SDK](https://cloud.google.com/sdk/docs/install) and
    [log in](https://cloud.google.com/sdk/gcloud/reference/auth/login):

    ```
    gcloud auth application-default login
    ```

1.  From your local machine use the
    [gcloud storage commands](https://cloud.google.com/storage/docs/gsuti https://cloud.google.com/sdk/gcloud/reference/storage)
    to transfer images in PNG format to the GCS bucket you set up in the
    previous step. If you have a large number of files to upload, you may
    consider using the
    [`rsync` command](https://cloud.google.com/sdk/gcloud/reference/storage/rsync)
    instead of `cp`.

    You should also include a path to a CSV file in gcs_metadata_csv. This CSV should contain a column with the file_names of the images you're uploading, titled by default 'img_id' and a label column for the task you want to train on, titled by default 'diagnostic'. We have set these titles as parameters however so you can adjust them if you like when you adjust the Demo Notebook to match your CSV.

1.  Make sure that [the email identity you selected](#how-to-gain-access) has
    the necessary permissions to view the images. The simplest method is to
    assign the predefined role of `roles/storage.objectViewer` to the chosen
    email identity. There are
    [several ways to do this](https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add).
    You should familiarize yourself with
    [GCS access control](https://cloud.google.com/storage/docs/access-control).

1.  Modify the [Demo Notebook](https://colab.research.google.com/github/Google-Health/imaging-research/blob/master/derm-foundation/derm_foundation_demo.ipynb#scrollTo=OxzYsc8NDpwa) and replace the values for:
    * gcp_project
    * gcs_bucket_name
    * gcs_metadata_csv
    * gcs_image_dir (leave blank if the images are in the root directory); and
    * label_column (the name of the columns for the label you're training for)
    * img_join_column (the name of the column you want to join your image files on)

    With the values from your GCS bucket.

    Also make sure you uncheck the "gcs_use_precomputed embeddings" flag.

## Access Options

You have the option to request access to the API either as
[an individual](#as-an-individual-non-gmail-account) or for [a group](#as-a-group-recommended).
Choose the process that best aligns with your needs. Remember to note the email
identifier for which you will be requesting access. It should be in one of these
formats:

*   YOUR-GROUP-NAME@YOUR-DOMAIN
*   INDIVIDUAL-ID@YOUR-DOMAIN
*   INDIVIDUAL-ID@gmail.com

### As a group (recommended)

If your organization is a Google Workspace or Google Cloud Platform (GCP)
customer, contact your Google admin and ask them to create a group with the list
of individuals who will be using the API. Let them know that this group is used
for contacting you and also as a security principal for authorizing your access
to the API.

![Create Google Group](img/create-group.png)

Otherwise,
[create a free Cloud Identity Account](https://cloud.google.com/identity/docs/set-up-cloud-identity-admin)
for your domain name and in the process become the interim Google admin for your
organization. Visit [Google Admin console](https://admin.google.com/) and create
the above-mentioned group. If your individual identities are unknown to Google,
they will need to follow the process for the [individuals](#as-an-individual)
before you can add them to the group.

### As an individual (non-gmail account)
This section applies for the INDIVIDUAL-ID@YOUR-DOMAIN case (e.g. `person@university.org` or `person@company.com`)

If your organization is a Google Workspace or GCP customer, identity federation
is most likely set up between your corporate identity directory and
[Google Identity and Access Management](https://cloud.google.com/security/products/iam)
and therefore individuals already have Google identities in the form of their
corporate emails. Check with your IT department to find out whether identity
federation is already in place or will be established soon.

Otherwise,
[create a Google identity based on your email](https://accounts.google.com/signup/v2/webcreateaccount?flowName=GlifWebSignIn&flowEntry=SignUp).
Opt for the "use my current email address instead" option, as shown in the
screen capture below.

IMPORTANT: You should choose a password that is different from your corporate
password.

![Create Google Id](img/create-identity.png)

### As an individual (`@gmail.com` account)

If you want to sign up as an individual with a gmail account, you can submit the form directly with your gmail address.


## General notes

*   Google does not keep a copy of any images sent.
*   Google monitors daily query volume and aggregates on a per-user and
    per-organization basis. Access can be revoked if a user or organization
    exceeds a reasonable query volume.

## Contributing

See [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) for details.

## License

See [`LICENSE`](LICENSE) for details.

## Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contact us

Please reach out to us at
[derm-foundation@google.com](mailto:derm-foundation@google.com]) for issues such
as, but not limited to:

-   Seeking technical assistance
-   Providing feedback
-   Requesting permissions for publications
-   Discussing clinical use cases
-   Discussing enterprise requirements such as:
    -   Fitting within strict security perimeters of your organization
    -   Governing your data in GCS
    -   Training and serving custom models at scale on
        [Vertex AI](https://cloud.google.com/vertex-ai?hl=en)

# Model Card for Derm Foundation

This tool uses an ML model to provide the embedding results. This section
briefly overviews the background and limitations of that model.

## Model Details

### Overview

This model generates embeddings for images of dermatological skin conditions.
Embeddings are n-dimensional vectors of floating points representing a
projection of the original image into a compressed feature space capable of
describing image features relevant to differentiating skin conditions and
properties (age, body part, etc.). These embeddings are to be used by
“downstream models” for final tasks such as condition category classification or
body part identification. The model uses the BiT-101x3 architecture
(https://arxiv.org/pdf/1912.11370.pdf). It was trained in two stages. The first
pre-training stage used contrastive learning to train on a large number of
public image-text pairs from the internet. The image component of this
pre-trained model was then fine-tuned for condition classification and a couple
other downstream tasks using a number of clinical datasets (see below).

Training Data:

*   Base model (pre-training): A large number of health-related image-text pairs
    from the public web
*   SFT (supervised fine-tuned) model: tele-dermatology datasets from the United
    States and Colombia, a skin cancer dataset from Australia, and additional
    public images. The images come from a mix of device types, including images
    from smartphone cameras, other cameras, and dermatoscopes. The images also
    have a mix of image takers; images may have been taken by clinicians during
    consultations or self-captured by patients.

### Version

```
name: v1.0.0
date: 2023-12-19
```

### Owners

```
derm-foundation@google.com
```

### Licenses

-   See
    [Derm Foundation - Additional Terms of Service](https://forms.gle/VBFuzSJXhQjNmF776).

### References

-   BiT: https://arxiv.org/pdf/1912.11370.pdf
-   CLIP: https://arxiv.org/abs/2103.00020

## Considerations

### Use Cases

-   Embeddings can reduce barriers to entry for training custom models for
    derm-specific tasks with less data, setup, and compute.
-   Embeddings can allow for quick evaluation.

### Limitations

-   The base model was trained using image-text pairs from the public web. These
    images come from a variety of sources but may by noisy or low-quality. The
    SFT (supervised fine-tuned) model was trained data from a limited set of
    countries (United States, Colombia, Australia, public images) and settings
    (mostly clinical). It may not generalize well to data from other countries,
    patient populations, or image types not used in training.
-   The model is only used to generate embeddings of the user-owned dataset. It
    does not generate any predictions or diagnosis on its own.
-   Developers should ensure any downstream model developed using this tool is
    validated to ensure performance is consistent against intended demographics
    e.g., skin tone, age, sex, gender etc.

### Ethical Considerations

-   Risk: Although Google does not store permanently any data sent to this
    model, it is the data owner's responsibility to ensure that Personally
    identifiable information (PII) and Protected Health Information (PHI) are
    removed prior to being sent to the model. \
-   Mitigation Strategy: Do not send data containing PII or PHI.
