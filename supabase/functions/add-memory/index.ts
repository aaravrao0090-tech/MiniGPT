import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405, headers: corsHeaders });

  try {
    const { content, passcode } = await req.json();
    const expected = Deno.env.get("DEVELOPER_PASSCODE");
    if (!expected || typeof passcode !== "string" || passcode !== expected) {
      return new Response(JSON.stringify({ error: "Incorrect passcode." }), {
        status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    if (typeof content !== "string" || content.trim().length < 10 || content.length > 5000) {
      return new Response(JSON.stringify({ error: "Information must be 10 to 5000 characters." }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const url = Deno.env.get("SUPABASE_URL");
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!url || !serviceKey) throw new Error("Server database configuration is missing.");

    const supabase = createClient(url, serviceKey);
    const { data, error } = await supabase.from("minigpt_memories")
      .insert({ content: content.trim() }).select("id, content, created_at").single();
    if (error) throw error;
    return new Response(JSON.stringify({ memory: data }), {
      status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (_error) {
    return new Response(JSON.stringify({ error: "Could not save shared memory. Check the server configuration." }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
