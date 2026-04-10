###############################################################################
# Outputs — GPU Node Pool
###############################################################################

output "node_pool_id" {
  description = "Azure resource ID of the GPU node pool."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.id
}

output "node_pool_name" {
  description = "Name of the GPU node pool."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.name
}

output "vm_size" {
  description = "VM SKU used for GPU nodes."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.vm_size
}

output "max_node_count" {
  description = "Maximum number of GPU nodes allowed by autoscaler."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.max_count
}

output "is_spot" {
  description = "Whether the node pool uses Spot VMs."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.priority == "Spot"
}
