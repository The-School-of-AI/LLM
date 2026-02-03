# AWS Resource Access Guidelines

This document outlines the standard operating procedure for requesting, provisioning, and managing AWS resources for teams across the charter.

---

## 📋 Process Overview

To ensure efficient resource management and budget control, please follow the steps below:

### 1. 📝 Submit Request
Teams requiring access to AWS resources must first complete the intake form:
- **Form:** [ERA V4 AWS Request Intake Form.xlsx](https://docs.google.com/spreadsheets/d/1n6QansCqeJ2vg02GPKfej1Vp4dpK7tao73xnMtouw98/edit?gid=0#gid=0)
- **Requirements:**
    - Fill in all template details comprehensively.
    - **Crucial:** Ensure "Halt Scripts" are ready to prevent budget overruns.

### 2. 🎫 Create Ticket
- Create an Issue in your team's GitHub project board. [Example - #269 ](https://github.com/The-School-of-AI/LLM/issues/269)
- Assign the issue to a **Team-13** member.

### 3. 🔍 Review (P-13)
- **P-13** will review the request.
- P-13 may contact you for clarification regarding infrastructure needs.

### 4. 💰 Budget Approval (P-14)
- Once reviewed by P-13, the request is assigned to **P-14**.
- P-14 reviews the proposed budget and assigns necessary credits.

### 5. 🔑 Provisioning
- Upon budget approval, **P-13** creates an IAM User with the specified permissions.
- Credentials are securely bonded to the requestor.

### 6. ⏳ Revocation
- On the **Expected End Date** (specified in the intake form), **P-13** will revoke the IAM User account.

---

## ⚠️ Important Notes


**1. Budget Monitoring & Alerts**
>
> To prevent unexpected costs, **P-13** will configure budget alerts for the requesting team. You will receive notifications at the following usage thresholds once you confirm via AWS automated email:
> - **50%** of budget
> - **75%** of budget
> - **90%** of budget

**2. Automatic Resource Termination**
> To ensure budget compliance and prevent cost overruns, **P-9** is authorized to automatically halt training sessions and shut down GPU resources.