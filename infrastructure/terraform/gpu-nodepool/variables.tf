###############################################################################
# Variables — GPU Node Pool
###############################################################################

variable "resource_group_name" {
  description = "Resource group containing the AKS cluster."
  type        = string
}

variable "aks_cluster_name" {
  description = "Name of the existing AKS cluster."
  type        = string
}

variable "node_pool_name" {
  description = "Name of the GPU node pool (max 12 chars, lowercase alphanumeric)."
  type        = string
  default     = "gpuinfer"

  validation {
    condition     = can(regex("^[a-z][a-z0-9]{0,11}$", var.node_pool_name))
    error_message = "Node pool name must be 1-12 lowercase alphanumeric characters starting with a letter."
  }
}

variable "gpu_vm_size" {
  description = "Azure VM SKU for the GPU nodes."
  type        = string
  default     = "Standard_NC6s_v3" # V100 16 GB

  validation {
    condition = contains([
      "Standard_NC6s_v3",          # 1x V100 16 GB
      "Standard_NC12s_v3",         # 2x V100 16 GB
      "Standard_NC24s_v3",         # 4x V100 16 GB
      "Standard_NC24ads_A100_v4",  # 1x A100 80 GB
      "Standard_NC48ads_A100_v4",  # 2x A100 80 GB
      "Standard_NC96ads_A100_v4",  # 4x A100 80 GB
    ], var.gpu_vm_size)
    error_message = "gpu_vm_size must be a supported NVIDIA GPU SKU."
  }
}

variable "min_node_count" {
  description = "Minimum number of nodes (0 = scale to zero)."
  type        = number
  default     = 0
}

variable "max_node_count" {
  description = "Maximum number of nodes for autoscaling."
  type        = number
  default     = 4
}

variable "enable_spot" {
  description = "Use Spot VMs for cost savings. Not recommended for latency-critical production."
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum price per hour for Spot VMs (-1 = on-demand cap)."
  type        = number
  default     = -1
}

variable "os_disk_size_gb" {
  description = "OS disk size in GB. Needs room for container images + model cache."
  type        = number
  default     = 128
}

variable "nvidia_plugin_version" {
  description = "Helm chart version for the NVIDIA device plugin."
  type        = string
  default     = "0.14.3"
}

variable "common_tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default = {
    "Project"     = "llmops-platform"
    "Environment" = "production"
  }
}
