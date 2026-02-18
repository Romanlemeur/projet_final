import Link from "next/link";
import AuthCard from "@/components/ui/AuthCard";
import Input from "@/components/ui/Input";
import SubmitButton from "@/components/ui/SubmitButton";

export const metadata = {
	title: "Set new password - Project",
};

export default function ResetPasswordPage() {
	return (
		<AuthCard>
			<form className="flex flex-col gap-4">
				<Input label="New password" type="password" id="password" name="password" placeholder="Enter new password" autoComplete="new-password" required />
				<Input label="Confirm password" type="password" id="confirm-password" name="confirm-password" placeholder="Confirm new password" autoComplete="new-password" required />
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
