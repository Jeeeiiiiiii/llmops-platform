###############################################################################
# Outputs — Storage
###############################################################################

output "storage_account_id" {
  description = "Resource ID of the storage account."
  value       = azurerm_storage_account.llmops.id
}

output "storage_account_name" {
  description = "Name of the storage account."
  value       = azurerm_storage_account.llmops.name
}

output "primary_blob_endpoint" {
  description = "Primary blob endpoint URL."
  value       = azurerm_storage_account.llmops.primary_blob_endpoint
}

output "model_weights_container" {
  description = "Name of the model weights container."
  value       = azurerm_storage_container.model_weights.name
}

output "vectordb_backups_container" {
  description = "Name of the vector DB backups container."
  value       = azurerm_storage_container.vectordb_backups.name
}

output "eval_artifacts_container" {
  description = "Name of the evaluation artifacts container."
  value       = azurerm_storage_container.eval_artifacts.name
}
