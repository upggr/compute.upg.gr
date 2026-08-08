# Coolify volume check (compute.upg.gr)

Verified on host `h2.buy-it.gr` for Coolify application uuid
`t4oocok4g804c0ogw0wooogg` (applicationId **8**, resource `upg-strings`).

## Mount

Docker Compose for the app includes:

```yaml
volumes:
  - 't4oocok4g804c0ogw0wooogg_static_data:/app/static/data'
```

Live container inspect confirms:

| Field | Value |
| --- | --- |
| Name | `t4oocok4g804c0ogw0wooogg_static_data` |
| Destination | `/app/static/data` |
| Driver | local |
| RW | true |

This persists `hall_of_fame.sqlite`, `geometry.sqlite`, `jobs.sqlite`, and
ephemeral `results_*.json` across redeploys. Agents must not trigger Coolify
redeploys manually; git push auto-builds via webhook.
