# Base on the most recently released Fedora
FROM fedora:latest
MAINTAINER ipbabble email buildahboy@redhat.com # not a real email

# Install updates and httpd
RUN echo "Updating all fedora packages"; dnf -y update; dnf -y clean all
RUN echo "Installing httpd"; dnf -y install httpd && dnf -y clean all

# Expose the default httpd port 80
EXPOSE 80

# Run the httpd
CMD ["/usr/sbin/httpd", "-DFOREGROUND"]
