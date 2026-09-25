import { redirect } from "next/navigation";

import { hasSession } from "@/lib/session";

export default async function Home() {
  redirect((await hasSession()) ? "/dashboard" : "/login");
}
