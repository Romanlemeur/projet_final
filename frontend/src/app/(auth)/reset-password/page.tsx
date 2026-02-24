"use client";

import Link from "next/link";
import { useActionState } from "react";
import AuthCard from "@/components/ui/AuthCard";
import Input from "@/components/ui/Input";
import SubmitButton from "@/components/ui/SubmitButton";
import { resetPassword } from "@/lib/auth-actions";

export default function ResetPasswordPage() {
	const [state, formAction] = useActionState(resetPassword, null);

	return (
		<AuthCard>
			<h1 className="text-2xl font-bold tracking-tight text-white mb-8">Set new password</h1>

			<form action={formAction} className="flex flex-col gap-4">
				<Input label="New password" type="password" id="password" name="password" placeholder="Enter new password" autoComplete="new-password" required />
				<Input label="Confirm password" type="password" id="confirm-password" name="confirm-password" placeholder="Confirm new password" autoComplete="new-password" required />

				{state !== null && "error" in state && <p className="text-sm text-red-400 text-center">{state.error}</p>}

				<SubmitButton>Reset password</SubmitButton>
			</form>

			<p className="text-sm text-white/50 text-center mt-6">
				<Link href="/login" className="text-white hover:text-green-400 transition-colors font-medium rounded">
					Back to sign in
				</Link>
			</p>
		</AuthCard>
	);
}
