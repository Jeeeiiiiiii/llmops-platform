###############################################################################
# Variables — Storage
###############################################################################

variable "resource_group_name" {
  description = "Resource group for storage resources."
  type        = string
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus2"
}

variable "storage_account_name" {
  description = "Globally unique storage account name (3-24 lowercase alphanumeric)."
  type        = string
  default     = "stllmopsmodels"

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "Storage account name must be 3-24 lowercase alphanumeric characters."
  }
}

variable "replication_type" {
  description = "Storage replication type."
  type        = string
  default     = "ZRS" # Zone-redundant for production

  validation {
    condition     = contains(["LRS", "ZRS", "GRS", "RAGRS"], var.replication_type)
    error_message = "Must be LRS, ZRS, GRS, or RAGRS."
  }
}

variable "enable_public_access" {
  description = "Allow public network access. Set false for production."
  type        = bool
  default     = false
}

variable "allowed_subnet_ids" {
  description = "Subnet IDs allowed through the storage firewall."
  type        = list(string)
  default     = []
}

variable "soft_delete_retention_days" {
  description = "Soft-delete retention in days."
  type        = number
  default     = 30
}

variable "common_tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default = {
    "Project"     = "llmops-platform"
    "Environment" = "production"
  }
}
