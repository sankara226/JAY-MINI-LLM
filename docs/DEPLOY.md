# Put Jay on the web for free

This guide puts the chat page online with **Hugging Face Spaces**, a free hosting service for machine-learning demos. When you're done, anyone can chat with Jay at a link like `https://huggingface.co/spaces/your-name/chat-with-jay`.

- **Cost:** free (the "CPU basic" plan: 2 CPU cores, 16 GB of memory, which is plenty for Jay).
- **Time:** about 20 minutes, plus 5–10 minutes of waiting while the Space builds.
- **You need:** a web browser, and the files from `MyDrive/tiny_llm/` on your computer (`tokenizer.json`, `chat_model.pt`, `chat_model_v2.pt`, `chat_model_v3.pt`).
- **No terminal needed:** everything is done by clicking and uploading.

## How it works (the big picture)

You'll create **two** things on Hugging Face:

| What | What it holds | Why |
|---|---|---|
| A **model repository** (`jay-weights`) | The trained files: `tokenizer.json` and the `.pt` weights | Hugging Face's home for model files. The weights stay separate from the code. |
| A **Space** (`chat-with-jay`) | The code: `app.py`, `jay/`, `static/`, `requirements.txt` | Runs the code and shows the web page. When it starts, it downloads the weights from the model repository. |

Keeping weights and code apart is standard practice: the code stays small, and you can update one without touching the other.

---

## Step 1: Create a Hugging Face account

1. Go to <https://huggingface.co/join> and sign up (it's free).
2. Confirm your email address.
3. Remember your **username**. It appears in your links, for example `huggingface.co/your-name`.

## Step 2: Upload the weights to a model repository

1. Click your profile picture (top right) → **New Model**.
2. Fill in:
   - **Model name:** `jay-weights`
   - **License:** pick one (for example `mit`), or leave it empty.
   - **Visibility:** **Public**, so the Space can download the files without a password.
3. Click **Create model**.
4. Open the **Files and versions** tab → **Add file** → **Upload files**.
5. Drag in these four files: `tokenizer.json`, `chat_model.pt`, `chat_model_v2.pt`, `chat_model_v3.pt`.
   - You can skip `chat_model.pt` or `chat_model_v2.pt`. The model picker only offers the versions it finds.
   - Don't upload `base_model.pt`: the chat page doesn't use it.
6. At the bottom, write a short message such as "Add Jay's weights", then click **Commit changes to main**. The upload takes a minute or two (about 150 MB).

Your weights now live at `https://huggingface.co/your-name/jay-weights`. The part after `huggingface.co/` (`your-name/jay-weights`) is the **repository id**. You'll need it in Step 5.

## Step 3: Create the Space

1. Click your profile picture → **New Space**.
2. Fill in:
   - **Space name:** `chat-with-jay`
   - **Select the Space SDK:** **Gradio**
   - **Space hardware:** **CPU basic · 2 vCPU · 16 GB · FREE**
   - **Visibility:** **Public**
3. Click **Create Space**. You land on an empty Space with a message about adding files.

## Step 4: Upload the code

1. In your Space, open the **Files** tab → **Add file** → **Upload files**.
2. From the project folder on your computer, drag in:
   - `app.py`
   - `requirements.txt`
   - the whole **`jay`** folder (`__init__.py`, `chat.py`, `model.py`)
   - the whole **`static`** folder (`jay.css`)

   Dragging a folder keeps its name, so the Space ends up with `jay/chat.py`, `static/jay.css` and so on. If your browser drops the folders, upload them one at a time: type `jay/` in the file-name box before choosing the files.
3. **Don't** upload the `.pt` files, the `weights` folder, `.venv`, the notebook or the `.zip` file.
4. Click **Commit changes to main**.

### Step 4b: Make the build faster (recommended)

The normal PyTorch download is about 2 GB, because it includes GPU support that a free CPU Space can't use. You can point it at the smaller CPU-only version (about 200 MB):

1. In the **Files** tab, click `requirements.txt` → **Edit**.
2. Add this line at the **very top**:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. Click **Commit changes to main**.

If the build later fails on `torch`, delete that line again: the Space still works, it just takes longer to build.

### Step 4c: Tell the Space which Python and Gradio versions to use

The Space has its own `README.md`, which starts with a block of settings between two `---` lines. Click `README.md` → **Edit** and make that block look like this (keep your own title):

```yaml
---
title: Chat with Jay
emoji: 💬
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
---
```

`sdk_version` must match the `gradio` version in `requirements.txt` (`6.31.0`). Click **Commit changes to main**.

## Step 5: Tell the Space where the weights are

The app reads the name of the weights repository from a setting called `JAY_MODEL_REPO`.

1. In your Space, open **Settings**.
2. Scroll to **Variables and secrets** → **New variable**.
3. Fill in:
   - **Name:** `JAY_MODEL_REPO`
   - **Value:** your repository id from Step 2, for example `your-name/jay-weights`
4. Click **Save**. The Space restarts by itself.

Use **New variable**, not **New secret**: the weights repository is public, so this value isn't a secret.

## Step 6: Wait for the build, then test

1. Open the **App** tab. You'll see **Building**, then **Running**. The first build takes 5–10 minutes.
2. To see what's happening, click **Logs** (next to the status). The lines that matter are pip installing the libraries, then Gradio printing `Running on local URL`.
3. When the status turns green (**Running**), the chat page appears. Try the example buttons:
   - "Who are you?" → Jay introduces itself.
   - "What is 4 + 3?" → the step-by-step counting answer.
   - "What is the capital of Japan?" → "I'm sorry, I don't know that…"
4. Share the link from your browser's address bar: `https://huggingface.co/spaces/your-name/chat-with-jay`.

---

## If something goes wrong

| What you see | What it means | Fix |
|---|---|---|
| `FileNotFoundError: No chat_model*.pt in …/weights` | The Space didn't download the weights | Check Step 5: the variable must be named exactly `JAY_MODEL_REPO`, and its value must be your exact repository id (`your-name/jay-weights`). The model repository must be **Public**. |
| `RepositoryNotFoundError` or `401` in the logs | The repository id is wrong, or the repository is private | Copy the id from the address bar of your model repository page. Make the repository public (Settings → Change visibility). |
| `ModuleNotFoundError: No module named 'jay'` | The `jay` folder wasn't uploaded as a folder | In the **Files** tab you must see `jay/chat.py`, not `chat.py` on its own. Re-upload into a `jay/` folder. |
| The page works but looks plain, with no rounded message box | `static/jay.css` is missing | Upload it into a `static/` folder. |
| The build fails while installing `torch` | Usually the line from Step 4b | Remove that line from `requirements.txt` and commit again. |
| The Space shows **Sleeping** | Free Spaces sleep after about 48 hours with no visitors | Open the link: it wakes up in about a minute. |
| Replies are slow when several people chat at once | The free plan has 2 CPU cores | Normal. Each reply takes a few seconds. |

## Updating the Space later

- **Code changes:** upload the changed file again in the Space's **Files** tab. The Space rebuilds by itself.
- **New weights** (for example a `chat_model_v4.pt`): upload them to the **model repository**, then restart the Space (**Settings** → **Factory reboot**). To show v4 in the model picker, `chat_model_v4.pt` must also be added to `MODEL_FILES` in `jay/chat.py`.

## Quicker but temporary: a share link from Colab

To show Jay to someone for a few days without setting up a Space, run the app in Google Colab with `share=True` (see the README). Gradio prints a public `https://….gradio.live` link. The link stops working when the Colab session ends.
