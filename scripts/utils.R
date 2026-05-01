notify_me_done <- function(subject = "✅ R script finished", to = "carsonslater7@gmail.com",
                           from = "carsonslater7@gmail.com", body = NULL, include_session_info = TRUE) {
  stopifnot(nzchar(Sys.getenv("SMTP_PASSWORD")))
  if (is.null(body)) {
    body <- paste0(
      "Done at: ", format(Sys.time()), "\n\n",
      if (include_session_info) {
        paste0(
          "User: ", Sys.info()[["user"]], "\n",
          "Host: ", Sys.info()[["nodename"]], "\n", "R:    ",
          R.version.string, "\n", "wd:   ", normalizePath(getwd(),
            winslash = "/", mustWork = FALSE
          ), "\n"
        )
      } else {
        ""
      }
    )
  }
  email_obj <- blastula::compose_email(body = blastula::md(paste0(
    "```\n",
    body, "\n```"
  )))
  blastula::smtp_send(
    email = email_obj, to = to, from = from,
    subject = subject, credentials = blastula::creds_envvar(
      user = from,
      pass_envvar = "SMTP_PASSWORD", provider = "gmail"
    )
  )
  invisible(TRUE)
}
