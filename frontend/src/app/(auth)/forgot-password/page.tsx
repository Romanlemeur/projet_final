"use client";

import Link from "next/link";
import { useActionState } from "react";
import AuthCard from "@/components/ui/AuthCard";
import Input from "@/components/ui/Input";
import SubmitButton from "@/components/ui/SubmitButton";
import { forgotPassword } from "@/lib/auth-actions";

export default function ForgotPasswordPage() {
	const [state, formAction] = useActionState(forgotPassword, null);

	const succeeded = state !== null && "success" in state;

	return (
		<AuthCard>
			{succeeded ? (
				<p className="text-sm text-green-400 text-center">Check your inbox - a password reset link has been sent.</p>
			) : (
				<form action={formAction} className="flex flex-col gap-4">
					<Input label="Email" type="email" id="email" name="email" placeholder="you@example.com" autoComplete="email" required />

					{state !== null && "error" in state && <p className="text-sm text-red-400 text-center">{state.error}</p>}

					<SubmitButton>Send reset link</SubmitButton>
				</form>
			)}

			<p className="text-sm text-white/50 text-center mt-6">
				<Link href="/login" className="text-white hover:text-green-400 transition-colors font-medium rounded">
					Back to sign in
				</Link>
			</p>
		</AuthCard>
	);
}
