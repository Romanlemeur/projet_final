"use server";

import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";

type ActionState = { error: string } | { success: true } | null;

export async function signUp(prevState: ActionState, formData: FormData): Promise<ActionState> {
	const supabase = await createClient();

	const email = formData.get("email") as string;
	const password = formData.get("password") as string;
	const confirmPassword = formData.get("confirm-password") as string;

	if (password !== confirmPassword) {
		return { error: "Passwords do not match." };
	}

	const { error } = await supabase.auth.signUp({ email, password });

	if (error) return { error: error.message };

	redirect("/dashboard");
}

export async function signIn(prevState: ActionState, formData: FormData): Promise<ActionState> {
	const supabase = await createClient();

	const email = formData.get("email") as string;
	const password = formData.get("password") as string;

	const { error } = await supabase.auth.signInWithPassword({ email, password });

	if (error) return { error: error.message };

	redirect("/dashboard");
}

export async function signOut() {
	const supabase = await createClient();
	await supabase.auth.signOut();
	redirect("/");
}

export async function forgotPassword(prevState: ActionState, formData: FormData): Promise<ActionState> {
	const supabase = await createClient();

	const email = formData.get("email") as string;
	const origin = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

	const { error } = await supabase.auth.resetPasswordForEmail(email, {
		redirectTo: `${origin}/auth/callback?next=/reset-password`,
	});

	if (error) return { error: error.message };

	return { success: true };
}

export async function resetPassword(prevState: ActionState, formData: FormData): Promise<ActionState> {
	const supabase = await createClient();

	const password = formData.get("password") as string;
	const confirmPassword = formData.get("confirm-password") as string;

	if (password !== confirmPassword) {
		return { error: "Passwords do not match." };
	}

	const { error } = await supabase.auth.updateUser({ password });

	if (error) return { error: error.message };

	redirect("/dashboard");
}
