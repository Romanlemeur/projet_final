"use client";

import Link from "next/link";
import { useActionState } from "react";
import AuthCard from "@/components/ui/AuthCard";
import SocialButton from "@/components/ui/SocialButton";
import Input from "@/components/ui/Input";
import SubmitButton from "@/components/ui/SubmitButton";
import { signUp } from "@/lib/auth-actions";

export default function SignupPage() {
	const [state, formAction] = useActionState(signUp, null);

	return (
		<AuthCard>
			<div className="flex flex-col gap-3 mb-6">
				<SocialButton provider="google" />
				<SocialButton provider="github" />
			</div>

			<div className="flex items-center gap-3 mb-6">
				<div className="flex-1 h-px bg-white/10" />
				<span className="text-sm text-white/40">or continue with email</span>
				<div className="flex-1 h-px bg-white/10" />
			</div>

			<form action={formAction} className="flex flex-col gap-4">
				<Input label="Full name" type="text" id="name" name="name" placeholder="Your full name" autoComplete="name" required />
				<Input label="Email" type="email" id="email" name="email" placeholder="you@example.com" autoComplete="email" required />
				<Input label="Password" type="password" id="password" name="password" placeholder="Create a password" autoComplete="new-password" required />
				<Input label="Confirm password" type="password" id="confirm-password" name="confirm-password" placeholder="Confirm your password" autoComplete="new-password" required />

				{state !== null && "error" in state && <p className="text-sm text-red-400 text-center">{state.error}</p>}

				<SubmitButton>Create account</SubmitButton>
			</form>

			<p className="text-sm text-white/50 text-center mt-6">
				Already have an account?{" "}
				<Link href="/login" className="text-white hover:text-green-400 transition-colors font-medium rounded">
					Sign in
				</Link>
			</p>
		</AuthCard>
	);
}
