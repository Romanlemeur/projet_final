import Link from "next/link";
import AuthCard from "@/components/ui/AuthCard";
import Input from "@/components/ui/Input";
import SubmitButton from "@/components/ui/SubmitButton";

export const metadata = {
	title: "Reset password - Project",
};

export default function ForgotPasswordPage() {
	return (
		<AuthCard>
			<form className="flex flex-col gap-4">
				<Input label="Email" type="email" id="email" name="email" placeholder="you@example.com" autoComplete="email" required />
				<SubmitButton>Send reset link</SubmitButton>
			</form>

			<p className="text-sm text-white/50 text-center mt-6">
				<Link href="/login" className="text-white hover:text-green-400 transition-colors font-medium rounded">
					Back to sign in
				</Link>
			</p>
		</AuthCard>
	);
}
