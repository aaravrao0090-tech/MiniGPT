# MiniGPT shared memory setup

The static site cannot safely store shared knowledge by itself. Shared memory uses a Supabase database plus a server-side Edge Function.

## 1. Create a Supabase project
Create a project at https://supabase.com. In Project Settings / API, copy the **Project URL** and the **anon/public key**. These two values can be used by the browser. Never put the service-role key in `index.html` or share it publicly.

## 2. Create the memory table
Open the Supabase SQL Editor and run the contents of `supabase/schema.sql`.

## 3. Deploy the secure save function
Install the Supabase CLI and sign in, then from this repository run:

```sh
supabase link --project-ref YOUR_PROJECT_REF
supabase secrets set 'DEVELOPER_PASSCODE=choose-a-long-private-passcode'
supabase functions deploy add-memory
```

Set the secret to the exact passcode you want to use. For production, choose a long, unique passcode and do not commit it to GitHub. The previously requested passcode should be considered exposed because it was shared in chat; use a new one for a real deployment.

## 4. Connect MiniGPT
Open the deployed website, choose **Developer mode**, enter the Supabase Project URL and anon/public key, and save the settings. Use **Load shared memory** to load facts saved by all users. To add a fact, enter the developer passcode and information, then click **Add to shared memory**.

## Notes
- Everyone connected to the same Supabase project reads the same shared memories.
- The mini n-gram model retrains on the facts loaded into its training text; this does not update the underlying PyTorch transformer weights.
- Anyone can read the shared memory, but only the server function should be able to write it. Keep Row Level Security enabled.
- This is a prototype. Shared facts are visible to all users, so do not store personal, private, or sensitive information.
