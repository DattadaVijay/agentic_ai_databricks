import mlflow

run_id = dbutils.widgets.get("run_id")
registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/databricks_governance_agent",
    name="databricks-governance-agent"
)

print(f"Name:    {registered.name}")
print(f"Version: {registered.version}")